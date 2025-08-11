# app.R — Shiny labeler with sparse co-occurrence & on-demand samplers
# --------------------------------------------------------------------
# Data layout assumed:
# - data/interim/keyword_candidates.csv    (kid, phrase, award_ids (comma string), omit/method/thematic logical)
# - data/interim/keyword_embeddings.parquet (kid + dim_000 ... dim_767)
# - kw_pairs.csv (optional; kw1, kw2, value) — appended as you label
#
# Samplers:
# - Uses keyword_samplers.R with on-demand sampling (no combn()).
# - Co-occur & hard-negative sample directly from sparse C.
#
# Notes:
# - used_key set is a character vector of "kidA||kidB" (unordered).
#   If no kw_pairs.csv exists, we pass NULL to samplers (skip check entirely).

library(shiny)
library(rhandsontable)
library(tidyverse)
library(arrow)
library(Matrix)

source("keyword_samplers.R")

pairs_file <- "../data/interim/kw_pairs.csv"

# Build an unordered key from two strings (A||B with lexical order)
pair_key <- function(a, b) paste(ifelse(a < b, a, b), ifelse(a < b, b, a), sep = "||")

# Unit-normalize rows (safe)
renorm <- function(M) {
  n <- sqrt(rowSums(M^2))
  n[n == 0] <- 1
  M / n
}

ui <- fluidPage(
  titlePanel("Keyword Pair Labeling Interface"),
  sidebarLayout(
    sidebarPanel(
      width=3,
      h3("Sampling Options"),
      selectInput(
        "sampler", "Choose a sampler:",
        choices = c(
          "Random" = "random",
          "Distance Midpoint" = "distance_mid",
          "Co-occurrence" = "cooccur",
          "Hard Negative" = "hard_negative"
        )
      ),
      numericInput("n_pairs", "Number of pairs to sample:", value = 10, min = 1, step = 1),
      
      # Sampler params
      conditionalPanel(
        condition = "input.sampler == 'distance_mid'",
        numericInput("target", "Target similarity:", value = 0.5, min = 0, max = 1, step = 0.05),
        numericInput("tol", "Tolerance:", value = 0.05, min = 0, max = 1, step = 0.01)
      ),
      conditionalPanel(
        condition = "input.sampler == 'cooccur'",
        numericInput("max_sim", "Max similarity:", value = 0.9, min = 0, max = 1, step = 0.05),
        numericInput("min_shared", "Minimum shared docs:", value = 1, min = 0, step = 1)
      ),
      conditionalPanel(
        condition = "input.sampler == 'hard_negative'",
        numericInput("min_sim", "Minimum similarity:", value = 0.7, min = 0, max = 1, step = 0.05),
        numericInput("max_shared_docs", "Max shared docs:", value = 0, min = 0, step = 1)
      ),
      
      actionButton("sample", "Sample Pairs")
    ),
    mainPanel(
      width=9,
      h3("Keyword Pairs to Label"),
      rHandsontableOutput("pair_table"),
      br(),
      fluidRow(
        column(4, actionButton("submit", "Submit")),
        column(4, actionButton("close", "Close"))
      )
    )
  )
)

server <- function(input, output, session) {
  # ----- Runtime state (scalars / matrices / frames) -----
  keyword_candidates <- NULL  # full candidates (data/interim/keyword_candidates.csv)
  kw_sub <- NULL              # active subset after filtering omit==TRUE
  emb_mat <- NULL             # all embeddings [rows keyed by kid]
  kw_sim <- NULL              # dense cosine for active subset
  C <- NULL                   # sparse co-occurrence for active subset (dgCMatrix, diag=0)
  kids <- NULL                # vector of kid for active subset (row names order)
  
  # Used pair keys (unordered "kid1||kid2"); NULL = skip check in samplers
  used_pair_keys <- reactiveVal(NULL)
  
  # Table to show for labeling
  sampled_df <- reactiveVal(NULL)
  
  # --------- Load data and initialize structures ----------
  tryCatch({
    # 1) Candidates
    keyword_candidates <- readr::read_csv("../data/interim/keyword_candidates.csv", show_col_types = FALSE) %>%
      mutate(
        kid       = as.character(kid),
        phrase    = as.character(phrase),
        award_ids = ifelse(is.na(award_ids), "", award_ids),
        award_ids = str_split(award_ids, ",\\s*"),
        omit      = ifelse(is.na(omit), FALSE, as.logical(omit)),
        method    = ifelse(is.na(method), FALSE, as.logical(method)),
        thematic  = ifelse(is.na(thematic), FALSE, as.logical(thematic))
      )
    
    # 2) Embeddings (keyword)
    kw_emb <- arrow::read_parquet("../data/interim/keyword_embeddings.parquet") %>%
      mutate(kid = as.character(kid)) %>%
      semi_join(keyword_candidates %>% select(kid), by = "kid") %>%
      arrange(kid)
    
    dim_cols <- grep("^dim_", names(kw_emb), value = TRUE)
    if (length(dim_cols) == 0) stop("No embedding columns 'dim_*' found in keyword_embeddings.parquet")
    emb_mat <- as.matrix(select(kw_emb, all_of(dim_cols)))
    rownames(emb_mat) <- kw_emb$kid
    emb_mat <- renorm(emb_mat)
    
    # 3) Active subset (omit==FALSE)
    kid_sub <- keyword_candidates %>% filter(!omit) %>% pull(kid)
    if (length(kid_sub) < 2) stop("After filtering omit==TRUE, not enough keywords remain.")
    kw_sub <- keyword_candidates %>% filter(kid %in% kid_sub) %>% arrange(match(kid, kid_sub))
    kids <- kw_sub$kid
    
    # Dense cosine for active subset
    emb_active <- emb_mat[kids, , drop = FALSE]
    kw_sim <- emb_active %*% t(emb_active)
    
    # 4) Sparse co-occurrence from award_ids
    all_docs <- sort(unique(unlist(kw_sub$award_ids)))
    if (length(all_docs) == 0) stop("award_ids empty after filtering.")
    rep_counts <- lengths(kw_sub$award_ids)
    rep_counts[is.na(rep_counts)] <- 0L
    
    i_idx <- rep(seq_len(nrow(kw_sub)), times = rep_counts)
    j_vals <- unlist(kw_sub$award_ids)
    doc_index <- setNames(seq_along(all_docs), all_docs)
    j_idx <- as.integer(unname(doc_index[j_vals]))
    keep <- !is.na(j_idx)
    i_idx <- i_idx[keep]; j_idx <- j_idx[keep]
    
    X <- sparseMatrix(i = i_idx, j = j_idx, x = 1L,
                      dims = c(nrow(kw_sub), length(all_docs)),
                      dimnames = list(kids, all_docs))
    X@x[] <- 1L
    C <- X %*% t(X)
    diag(C) <- 0
    C <- drop0(C)
    
    # 5) Used pair keys (from kw_pairs.csv), or NULL if no file
    if (file.exists(pairs_file)) {
      kp <- readr::read_csv(pairs_file,
                            col_names = c("kw1","kw2","value"),
                            show_col_types = FALSE) %>%
        filter(!is.na(kw1), !is.na(kw2))
      if (nrow(kp)) {
        # map phrases -> kids (for robustness, but phrases should be unique)
        phr_to_kid <- setNames(keyword_candidates$kid, keyword_candidates$phrase)
        a <- unname(phr_to_kid[kp$kw1])
        b <- unname(phr_to_kid[kp$kw2])
        keep_ab <- !is.na(a) & !is.na(b)
        if (any(keep_ab)) {
          used_pair_keys(unique(pair_key(a[keep_ab], b[keep_ab])))
        } else {
          used_pair_keys(NULL)
        }
      } else {
        used_pair_keys(NULL)
      }
    } else {
      used_pair_keys(NULL)
    }
    
  }, error = function(e) {
    showModal(modalDialog(
      title = "Error Loading Data",
      paste("Failed to load input files:", e$message),
      easyClose = TRUE
    ))
    return(invisible(NULL))
  })
  
  # ---------------------- Sampling -----------------------
  observeEvent(input$sample, {
    req(!is.null(kw_sub), !is.null(kw_sim), !is.null(C), !is.null(kids))
    
    n_kw <- nrow(kw_sub)
    if (n_kw < 2) {
      showNotification("Not enough active keywords to sample.", type = "error")
      return()
    }
    
    want <- input$n_pairs
    upk <- used_pair_keys()  # NULL means skip used-checks in samplers
    
    # Choose sampler (all return indices in current active order)
    idx <-
      if (input$sampler == "random") {
        sample_pair_random_n(n_kw = n_kw,
                             n = want,
                             kids = kids,
                             used_pair_keys = upk)
      } else if (input$sampler == "distance_mid") {
        sample_pair_distance_mid_n(sim = kw_sim,
                                   n = want,
                                   kids = kids,
                                   target = input$target,
                                   tol = input$tol,
                                   used_pair_keys = upk)
      } else if (input$sampler == "cooccur") {
        sample_pair_cooccur_sparse_n(n = want,
                                     C = C,
                                     sim = kw_sim,
                                     kids = kids,
                                     min_shared = input$min_shared,
                                     max_sim = input$max_sim,
                                     used_pair_keys = upk)
      } else if (input$sampler == "hard_negative") {
        sample_pair_hard_negative_sparse_n(n = want,
                                           C = C,
                                           sim = kw_sim,
                                           kids = kids,
                                           min_sim = input$min_sim,
                                           max_shared = input$max_shared_docs,
                                           used_pair_keys = upk)
      } else {
        matrix(NA_integer_, 0, 2)
      }
    
    if (!is.matrix(idx) || nrow(idx) == 0) {
      showNotification("No eligible pairs found. Try different parameters.", type = "warning")
      sampled_df(NULL); return()
    }
    
    phrase <- kw_sub$phrase
    df <- data.frame(
      `keyword 1`  = phrase[idx[, 1]],
      `omit 1`     = FALSE,
      `method 1`   = FALSE,
      `thematic 1` = FALSE,
      `keyword 2`  = phrase[idx[, 2]],
      `omit 2`     = FALSE,
      `method 2`   = FALSE,
      `thematic 2` = FALSE,
      value        = NA_real_,
      stringsAsFactors = FALSE
    )
    sampled_df(df)
  })
  
  output$pair_table <- renderRHandsontable({
    if (!is.null(sampled_df())) {
      rhandsontable(sampled_df(), useTypes = TRUE, stretchH = "all")
    }
  })
  
  # -------------------- Submit / Save --------------------
  observeEvent(input$submit, {
    submit_labeled_pairs()
  })
  
  observeEvent(input$close, { stopApp() })
  
  submit_labeled_pairs <- function() {
    df_edit <- hot_to_r(input$pair_table)
    if (is.null(df_edit) || nrow(df_edit) == 0) return()
    
    # 1) Update keyword metadata (by phrase)
    for (i in seq_len(nrow(df_edit))) {
      kw1 <- df_edit$`keyword.1`[i]
      kw2 <- df_edit$`keyword.2`[i]
      idx1 <- which(keyword_candidates$phrase == kw1)
      idx2 <- which(keyword_candidates$phrase == kw2)
      if (length(idx1) == 1) {
        if (!is.na(df_edit$`omit.1`[i]))     keyword_candidates$omit[idx1]     <<- as.logical(df_edit$`omit.1`[i])
        if (!is.na(df_edit$`method.1`[i]))   keyword_candidates$method[idx1]   <<- as.logical(df_edit$`method.1`[i])
        if (!is.na(df_edit$`thematic.1`[i])) keyword_candidates$thematic[idx1] <<- as.logical(df_edit$`thematic.1`[i])
      }
      if (length(idx2) == 1) {
        if (!is.na(df_edit$`omit.2`[i]))     keyword_candidates$omit[idx2]     <<- as.logical(df_edit$`omit.2`[i])
        if (!is.na(df_edit$`method.2`[i]))   keyword_candidates$method[idx2]   <<- as.logical(df_edit$`method.2`[i])
        if (!is.na(df_edit$`thematic.2`[i])) keyword_candidates$thematic[idx2] <<- as.logical(df_edit$`thematic.2`[i])
      }
    }
    
    # 2) Append labeled rows to kw_pairs.csv (only finite values)
    new_rows <- df_edit %>%
      filter(is.finite(value)) %>%
      transmute(kw1 = `keyword.1`, kw2 = `keyword.2`, value)
    if (nrow(new_rows)) {
      if (!file.exists(pairs_file)) {
        readr::write_csv(new_rows, pairs_file, quote='all')
      } else {
        readr::write_csv(new_rows, pairs_file, append = TRUE, quote='all')
      }
      
      # Update used_pair_keys with newly added pairs (by kid)
      phr_to_kid <- setNames(keyword_candidates$kid, keyword_candidates$phrase)
      a <- unname(phr_to_kid[new_rows$kw1])
      b <- unname(phr_to_kid[new_rows$kw2])
      keep_ab <- !is.na(a) & !is.na(b)
      if (any(keep_ab)) {
        added <- pair_key(a[keep_ab], b[keep_ab])
        cur <- used_pair_keys()
        if (is.null(cur)) cur <- character(0)
        used_pair_keys(unique(c(cur, added)))
      }
    }
    
    # 3) Persist updated candidates (award_ids back to comma string)
    keyword_candidates_out <- keyword_candidates %>%
      mutate(
        award_ids = vapply(award_ids, function(x) paste(x, collapse = ","), character(1)),
        omit      = ifelse(is.na(omit), FALSE, omit),
        method    = ifelse(is.na(method), FALSE, method),
        thematic  = ifelse(is.na(thematic), FALSE, thematic)
      )
    readr::write_csv(keyword_candidates_out, "../data/interim/keyword_candidates.csv", quote='all')
    
    # 4) If omits changed, refresh active subset, kw_sim, and C
    kid_sub_new <- keyword_candidates %>% filter(!omit) %>% pull(kid)
    if (!identical(sort(kid_sub_new), sort(kids))) {
      kw_sub_new <- keyword_candidates %>% filter(kid %in% kid_sub_new) %>% arrange(match(kid, kid_sub_new))
      kids <<- kw_sub_new$kid
      kw_sub <<- kw_sub_new
      
      # recompute sim for active subset
      emb_active <- emb_mat[kids, , drop = FALSE]
      kw_sim <<- emb_active %*% t(emb_active)
      
      # rebuild sparse C
      all_docs <- sort(unique(unlist(kw_sub$award_ids)))
      rep_counts <- lengths(kw_sub$award_ids); rep_counts[is.na(rep_counts)] <- 0L
      i_idx <- rep(seq_len(nrow(kw_sub)), times = rep_counts)
      j_vals <- unlist(kw_sub$award_ids)
      doc_index <- setNames(seq_along(all_docs), all_docs)
      j_idx <- as.integer(unname(doc_index[j_vals]))
      keep <- !is.na(j_idx)
      i_idx <- i_idx[keep]; j_idx <- j_idx[keep]
      
      X <- sparseMatrix(i = i_idx, j = j_idx, x = 1L,
                        dims = c(nrow(kw_sub), length(all_docs)),
                        dimnames = list(kids, all_docs))
      X@x[] <- 1L
      C_new <- X %*% t(X); diag(C_new) <- 0
      C <<- drop0(C_new)
    }
    
    showNotification("Saved.", type = "message")
  }
}

shinyApp(ui, server)
