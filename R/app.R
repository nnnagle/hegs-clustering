# app.R — Shiny labeler with sparse co-occurrence & on-demand samplers + abstract preview navigator
# ----------------------------------------------------------------------------------------------
# Data layout assumed:
# - ../data/interim/keyword_candidates.csv     (kid, phrase, award_ids (comma string), omit/method/thematic logical)
# - ../data/interim/keyword_embeddings.parquet (kid + dim_000 ... dim_767)
# - ../data/interim/kw_pairs.csv (optional; kw1, kw2, value, rater) — appended as you label
# - ../data/interim/HEGS_clean_df.csv          (award_id, abstract)
#
# Samplers:
# - Uses keyword_samplers.R with on-demand sampling (no combn()).
# - Co-occur & hard-negative sample directly from sparse C.
#
# Notes:
# - used_key set is a character vector of "kidA||kidB" (unordered) for the CURRENT rater.
#   If no kw_pairs.csv exists, we pass NULL to samplers (skip check entirely).

library(shiny)
library(shinyjs)
library(rhandsontable)
library(tidyverse)
library(arrow)
library(Matrix)
library(DBI)
library(RSQLite)

source("keyword_samplers.R")

pairs_file <- "../data/interim/kw_pairs.csv"

# Build an unordered key from two strings (A||B with lexical order)
pair_key <- function(a, b) paste(ifelse(a < b, a, b), ifelse(a < b, b, a), sep = "||")

# Safe helper: coalesce to empty string
nzchr <- function(x) if (is.null(x) || length(x) == 0 || is.na(x)) "" else as.character(x)

# Normalizer for rater initials (vector-safe)
norm_init <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  toupper(trimws(x))
}

# Unit-normalize rows (safe)
renorm <- function(M) {
  n <- sqrt(rowSums(M^2))
  n[n == 0] <- 1
  M / n
}

# Escape a phrase for regex, then wrap matches with <mark>
highlight_phrase <- function(text, phrase) {
  if (is.na(text) || is.na(phrase) || phrase == "") return(text)
  pat <- stringr::regex(stringr::str_escape(phrase), ignore_case = TRUE)
  locs <- stringr::str_locate_all(text, pat)[[1]]
  if (is.null(locs) || nrow(locs) == 0) return(text)
  out <- character(0)
  last <- 1L
  for (i in seq_len(nrow(locs))) {
    s <- locs[i,1]; e <- locs[i,2]
    out <- c(out, substr(text, last, s - 1L), "<mark>", substr(text, s, e), "</mark>")
    last <- e + 1L
  }
  out <- c(out, substr(text, last, nchar(text)))
  paste0(out, collapse = "")
}

ui <- fluidPage(
  useShinyjs(),
  titlePanel("Keyword Pair Labeling Interface"),
  tags$head(tags$style(HTML("
    mark {
      background-color: #008000 !important;
      color: #fff;
    }
  "))),
  sidebarLayout(
    sidebarPanel(
      width = 3,
      # === Rater initials + explicit load button ===
      textInput("rater", "Enter your initials here", value = "", placeholder = "e.g., AB"),
      actionButton("enter_rater", "Enter rater / refresh"),
      tags$hr(),
      
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
      
      actionButton("sample", "Sample Pairs"),
      
      div(style = "border-top: 2px solid #e5e7eb; margin-top: 18px; padding-top: 14px;",
          h3("Abstract Preview"),
          selectizeInput("kw_for_preview", "Select a keyword:", choices = NULL, selected = NULL,
                         options = list(placeholder = "Type to search…")),
          actionButton("preview_abs", "Show example abstract")
      )
    ),
    mainPanel(
      width = 9,
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
  # ----- Runtime state -----
  keyword_candidates <- NULL
  kw_sub <- NULL
  emb_mat <- NULL
  
  kids_universe <- NULL
  kw_sim_full <- NULL
  C_full <- NULL
  kids <- NULL
  
  used_pair_keys <- reactiveVal(NULL)
  kp_all <- reactiveVal(NULL)
  kp_by_rater <- reactiveVal(NULL)
  phr_to_kid <- reactiveVal(NULL)
  sampled_df <- reactiveVal(NULL)
  modal_state <- reactiveValues(ids = character(0), idx = 1, kw = NULL)
  
  data_ready <- reactiveVal(FALSE)
  
  # NEW: committed rater + loaded flag
  current_rater_input <- reactive(norm_init(input$rater))   # what’s typed right now
  entered_rater <- reactiveVal("")                          # committed on button click
  rater_loaded <- reactiveVal(FALSE)
  
  # Gate Sample strictly: base data must be ready AND rater must be entered
  observe({
    if (isTRUE(data_ready()) && isTRUE(rater_loaded())) {
      shinyjs::enable("sample")
    } else {
      shinyjs::disable("sample")
    }
  })
  # If initials text changes, require re-load
  observeEvent(input$rater, {
    rater_loaded(FALSE)
    shinyjs::disable("sample")
  }, ignoreInit = TRUE)
  
  # DB connection
  kc_db <- "../data/interim/keyword_candidates.sqlite"
  kc_con <- DBI::dbConnect(RSQLite::SQLite(), kc_db)
  
  # --------- Load data and initialize structures ----------
  hegs_df <- tryCatch({
    readr::read_csv("../data/interim/HEGS_clean_df.csv", show_col_types = FALSE) %>%
      mutate(award_id = as.character(award_id))
  }, error = function(e) {
    tibble(award_id = character(0), abstract = character(0))
  })
  
  keyword_candidates <- tryCatch({
    onStop(function() { try(DBI::dbDisconnect(kc_con), silent = TRUE) })
    invisible(DBI::dbExecute(kc_con, "PRAGMA journal_mode=WAL;"))
    invisible(DBI::dbExecute(kc_con, "PRAGMA synchronous=NORMAL;"))
    DBI::dbGetQuery(
      kc_con,
      "
      SELECT
        kid,
        phrase,
        COALESCE(award_ids, '') AS award_ids,
        COALESCE(omit,   0)     AS omit,
        COALESCE(method, 0)     AS method,
        COALESCE(thematic,0)    AS thematic
      FROM keyword_candidates
      "
    ) %>%
      mutate(
        kid       = as.character(kid),
        phrase    = as.character(phrase),
        award_ids = stringr::str_split(award_ids, ",\\s*"),
        omit      = as.logical(as.integer(omit)),
        method    = as.logical(as.integer(method)),
        thematic  = as.logical(as.integer(thematic))
      )
  }, error = function(e) {
    showModal(modalDialog(
      title = "Error Loading keyword candidates Data",
      paste("Failed to load input files:", e$message),
      easyClose = TRUE
    ))
    return(invisible(NULL))
  })
  
  phr_to_kid(setNames(keyword_candidates$kid, keyword_candidates$phrase))
  
  emb_mat <- tryCatch({
    kw_emb <- arrow::read_parquet("../data/interim/keyword_embeddings.parquet") %>%
      mutate(kid = as.character(kid)) %>%
      semi_join(keyword_candidates %>% dplyr::select(kid), by = "kid") %>%
      arrange(kid)
    dim_cols <- grep("^dim_", names(kw_emb), value = TRUE)
    if (length(dim_cols) == 0) stop("No embedding columns 'dim_*' found in keyword_embeddings.parquet")
    emb_mat_tmp <- as.matrix(select(kw_emb, all_of(dim_cols)))
    rownames(emb_mat_tmp) <- kw_emb$kid
    renorm(emb_mat_tmp)
  }, error = function(e) {
    showModal(modalDialog(
      title = "Error Loading Keyword Embeddings Data",
      paste("Failed to load input files:", e$message),
      easyClose = TRUE
    ))
    return(invisible(NULL))
  })
  
  kids_universe <- keyword_candidates$kid %>% as.character()
  kid_sub <- keyword_candidates %>% filter(!omit) %>% pull(kid)
  if (length(kid_sub) < 2) stop("After filtering omit==TRUE, less than two keywords remain.")
  kw_sub <- keyword_candidates %>%
    filter(kid %in% kid_sub) %>%
    arrange(match(kid, kid_sub))
  kids <- kw_sub$kid
  kw_sim_full <- emb_mat[kids,] %*% t(emb_mat[kids,])
  
  # Sparse co-occurrence
  all_docs <- sort(unique(unlist(kw_sub$award_ids)))
  doc_ids <- setNames(seq_along(all_docs), all_docs)
  kw_ids <- setNames(seq_along(kw_sub$kid), kw_sub$kid)
  i <- rep.int(x = kw_ids, times = lengths(kw_sub$award_ids))
  j <- doc_ids[unlist(kw_sub$award_ids)]
  X <- sparseMatrix(i = i, j = j, x = 1L,
                    dims = c(nrow(kw_sub), length(all_docs)),
                    dimnames = list(kw_sub$kid, all_docs))
  C_full <- X %*% t(X)
  diag(C_full) <- 0
  C_full <- drop0(C_full)
  
  tryCatch({
    # Do NOT load kw_pairs until "Enter rater"
    kp_all(NULL); kp_by_rater(NULL); used_pair_keys(NULL)
    
    updateSelectizeInput(session, "kw_for_preview",
                         choices = kw_sub$phrase,
                         selected = NULL,
                         server = TRUE)
    data_ready(TRUE)
  }, error = function(e) {
    showModal(modalDialog(
      title = "Error Loading Data",
      paste("Failed to load input files:", e$message),
      easyClose = TRUE
    ))
    return(invisible(NULL))
  })
  
  # Helper: compute used keys
  compute_used_keys <- function(rows_df) {
    if (is.null(rows_df) || nrow(rows_df) == 0) return(NULL)
    map <- phr_to_kid()
    a <- unname(map[rows_df$kw1]); b <- unname(map[rows_df$kw2])
    keep <- !is.na(a) & !is.na(b)
    if (!any(keep)) return(NULL)
    unique(pair_key(a[keep], b[keep]))
  }
  
  # Load/reload pairs for the current rater on button
  observeEvent(input$enter_rater, {
    rt <- current_rater_input()
    if (!nzchar(rt)) {
      used_pair_keys(NULL); kp_by_rater(NULL)
      rater_loaded(FALSE)
      showNotification("Enter your initials first.", type = "error")
      return()
    }
    
    # Commit the rater and reflect it in the text box (uppercased/trimmed)
    entered_rater(rt)
    updateTextInput(session, "rater", value = rt)
    
    if (!file.exists(pairs_file)) {
      kp_all(tibble(kw1=character(), kw2=character(), value=numeric(), rater=character()))
      kp_by_rater(NULL)
      used_pair_keys(NULL)
      rater_loaded(TRUE)
      showNotification("No kw_pairs.csv yet. Sampling will ignore used-pair checks.", type = "message")
      return()
    }
    
    ok <- TRUE
    kp <- tryCatch(readr::read_csv(pairs_file, show_col_types = FALSE), error = function(e) { ok <<- FALSE; NULL })
    if (!ok) {
      rater_loaded(FALSE)
      shinyjs::disable("sample")
      showNotification("Failed to read kw_pairs.csv.", type = "error")
      return()
    }
    
    if (!("kw1" %in% names(kp) && "kw2" %in% names(kp))) {
      if (ncol(kp) >= 3) names(kp)[1:3] <- c("kw1","kw2","value")
    }
    if (!("rater" %in% names(kp))) kp$rater <- NA_character_
    kp_all(kp)
    
    kp_r <- kp %>%
      mutate(rater = norm_init(rater)) %>%
      filter(rater == rt, !is.na(kw1), !is.na(kw2))
    
    kp_by_rater(kp_r)
    used_pair_keys(compute_used_keys(kp_r))
    rater_loaded(TRUE)
    showNotification(paste0("Loaded ", nrow(kp_r), " pairs for rater ", rt, "."), type = "message")
  }, ignoreInit = TRUE)
  
  # ---------------------- Sampling -----------------------
  # Reassert the committed rater into the text box on Sample click (defensive against UI wipes)
  observeEvent(input$sample, {
    updateTextInput(session, "rater", value = isolate(entered_rater()))
  }, ignoreInit = TRUE, priority = 100)
  
  observeEvent(input$sample, {
    req(!is.null(kw_sub), !is.null(kw_sim_full), !is.null(C_full), !is.null(kids_universe), !is.null(kids))
    if (!isTRUE(rater_loaded())) {
      showNotification("Click 'Enter rater / refresh' first.", type = "error"); return()
    }
    
    n_kw <- length(kids)
    if (n_kw < 2) { showNotification("Not enough active keywords to sample.", type = "error"); return() }
    
    idx <- match(kids, kids_universe)
    sim_active <- kw_sim_full[kids, kids, drop = FALSE]
    C_active   <- C_full[kids,      kids,  drop = FALSE]
    
    want <- input$n_pairs
    upk <- used_pair_keys()
    
    idx_pairs <-
      if (input$sampler == "random") {
        sample_pair_random_n(n_kw = n_kw, n = want, kids = kids, used_pair_keys = upk)
      } else if (input$sampler == "distance_mid") {
        sample_pair_distance_mid_n(sim = sim_active, n = want, kids = kids,
                                   target = input$target, tol = input$tol, used_pair_keys = upk)
      } else if (input$sampler == "cooccur") {
        sample_pair_cooccur_sparse_n(n = want, C = C_active, sim = sim_active, kids = kids,
                                     min_shared = input$min_shared, max_sim = input$max_sim, used_pair_keys = upk)
      } else if (input$sampler == "hard_negative") {
        sample_pair_hard_negative_sparse_n(n = want, C = C_active, sim = sim_active, kids = kids,
                                           min_sim = input$min_sim, max_shared = input$max_shared_docs, used_pair_keys = upk)
      } else {
        matrix(NA_integer_, 0, 2)
      }
    
    if (!is.matrix(idx_pairs) || nrow(idx_pairs) == 0) {
      showNotification("No eligible pairs found. Try different parameters.", type = "warning")
      sampled_df(NULL); return()
    }
    
    phrase <- kw_sub$phrase
    df <- tibble(
      `keyword 1`  = phrase[idx_pairs[, 1]],
      `omit 1`     = FALSE,
      `method 1`   = FALSE,
      `thematic 1` = FALSE,
      `keyword 2`  = phrase[idx_pairs[, 2]],
      `omit 2`     = FALSE,
      `method 2`   = FALSE,
      `thematic 2` = FALSE,
      value        = NA_real_
    )
    sampled_df(df)
  })
  
  output$pair_table <- renderRHandsontable({
    if (!is.null(sampled_df())) rhandsontable(sampled_df(), useTypes = TRUE, stretchH = "all")
  })
  
  # -------------------- Submit / Save --------------------
  observeEvent(input$submit, {
    shinyjs::disable("submit")
    on.exit(shinyjs::enable("submit"), add = TRUE)
    submit_labeled_pairs()
  })
  
  observeEvent(input$close, { stopApp() })
  
  submit_labeled_pairs <- function() {
    df_edit <- hot_to_r(input$pair_table)
    if (is.null(df_edit) || nrow(df_edit) == 0) return()
    
    rt <- entered_rater()   # use the committed rater, not the raw input
    if (!nzchar(rt)) { showNotification("Enter your initials before submitting.", type = "error"); return() }
    
    # 1) Update keyword metadata (by phrase) in-memory
    for (i in seq_len(nrow(df_edit))) {
      kw1 <- df_edit$`keyword 1`[i]
      kw2 <- df_edit$`keyword 2`[i]
      idx1 <- which(keyword_candidates$phrase == kw1)
      idx2 <- which(keyword_candidates$phrase == kw2)
      if (length(idx1) == 1) {
        if (!is.na(df_edit$`omit 1`[i]))     keyword_candidates$omit[idx1]     <<- as.logical(df_edit$`omit 1`[i])
        if (!is.na(df_edit$`method 1`[i]))   keyword_candidates$method[idx1]   <<- as.logical(df_edit$`method 1`[i])
        if (!is.na(df_edit$`thematic 1`[i])) keyword_candidates$thematic[idx1] <<- as.logical(df_edit$`thematic 1`[i])
      }
      if (length(idx2) == 1) {
        if (!is.na(df_edit$`omit 2`[i]))     keyword_candidates$omit[idx2]     <<- as.logical(df_edit$`omit 2`[i])
        if (!is.na(df_edit$`method 2`[i]))   keyword_candidates$method[idx2]   <<- as.logical(df_edit$`method 2`[i])
        if (!is.na(df_edit$`thematic 2`[i])) keyword_candidates$thematic[idx2] <<- as.logical(df_edit$`thematic 2`[i])
      }
    }
    
    # 2) Append labeled rows to kw_pairs.csv (only finite values), with rater column
    new_rows <- df_edit %>%
      dplyr::filter(is.finite(value)) %>%
      dplyr::transmute(kw1 = `keyword 1`, kw2 = `keyword 2`, value, rater = rt)
    
    if (nrow(new_rows)) {
      new_rows <- new_rows %>% dplyr::select(kw1, kw2, value, rater)
      
      if (!file.exists(pairs_file)) {
        readr::write_csv(new_rows, pairs_file, quote = "all")
      } else {
        kp <- tryCatch(readr::read_csv(pairs_file, show_col_types = FALSE), error = function(e) NULL)
        if (is.null(kp)) {
          readr::write_csv(new_rows, pairs_file, append = TRUE, quote = "all")
        } else {
          if (!("kw1" %in% names(kp) && "kw2" %in% names(kp))) {
            if (ncol(kp) >= 3) names(kp)[1:3] <- c("kw1","kw2","value")
          }
          if (!("rater" %in% names(kp))) kp$rater <- NA_character_
          kp <- kp %>% dplyr::select(kw1, kw2, value, rater)
          kp <- dplyr::bind_rows(kp, new_rows)
          readr::write_csv(kp, pairs_file, quote = "all")
        }
      }
      
      # Update in-memory stores and used_pair_keys for this rater
      cur_all <- kp_all()
      if (is.null(cur_all)) {
        kp_all(new_rows)
      } else {
        cur_all <- cur_all %>%
          { if (!("rater" %in% names(.))) mutate(., rater = NA_character_) else . } %>%
          { if (!("kw1" %in% names(.))) { names(.)[1:3] <- c("kw1","kw2","value"); . } else . } %>%
          dplyr::select(kw1, kw2, value, rater)
        kp_all(bind_rows(cur_all, new_rows))
      }
      
      nr_r <- new_rows %>% mutate(rater = norm_init(rater)) %>% filter(rater == rt)
      if (nrow(nr_r)) {
        old_r <- kp_by_rater(); if (is.null(old_r)) old_r <- nr_r[0,]
        kp_by_rater(bind_rows(old_r, nr_r))
        added <- compute_used_keys(nr_r)
        cur <- used_pair_keys(); if (is.null(cur)) cur <- character(0)
        used_pair_keys(unique(c(cur, added)))
      }
    }
    
    # 3) Persist updated candidates to SQLite (flags only)
    phr2kid <- phr_to_kid()
    kids_touched <- unique(c(df_edit$`keyword 1`, df_edit$`keyword 2`))
    kids_touched <- unname(phr2kid[kids_touched])
    kids_touched <- kids_touched[!is.na(kids_touched)]
    
    if (length(kids_touched)) {
      upd_df <- keyword_candidates %>%
        filter(kid %in% kids_touched) %>%
        transmute(
          kid,
          omit     = as.integer(omit),
          method   = as.integer(method),
          thematic = as.integer(thematic)
        )
      tryCatch({
        DBI::dbWithTransaction(kc_con, {
          DBI::dbExecute(kc_con, "DROP TABLE IF EXISTS tmp_updates")
          DBI::dbWriteTable(kc_con, "tmp_updates", upd_df, temporary = TRUE)
          DBI::dbExecute(kc_con, "
            UPDATE keyword_candidates
               SET omit     = (SELECT omit     FROM tmp_updates u WHERE u.kid = keyword_candidates.kid),
                   method   = (SELECT method   FROM tmp_updates u WHERE u.kid = keyword_candidates.kid),
                   thematic = (SELECT thematic FROM tmp_updates u WHERE u.kid = keyword_candidates.kid)
             WHERE kid IN (SELECT kid FROM tmp_updates)
          ")
        })
      }, error = function(e) {
        showNotification(paste("SQLite update failed:", e$message), type = "error")
      })
    }
    
    # 4) Update active subset ONLY
    kid_sub_new <- keyword_candidates %>% filter(!omit) %>% pull(kid)
    if (!identical(kid_sub_new, kids)) {
      kw_sub_new <- keyword_candidates %>%
        filter(kid %in% kid_sub_new) %>%
        arrange(match(kid, kid_sub_new))
      kids <<- kw_sub_new$kid
      kw_sub <<- kw_sub_new
      updateSelectizeInput(session, "kw_for_preview",
                           choices = kw_sub$phrase,
                           selected = NULL,
                           server = TRUE)
    }
    
    if (length(kids) < 2) data_ready(FALSE) else data_ready(TRUE)
    showNotification("Saved.", type = "message")
  }
  
  # ------------------ Abstract preview modal ------------------
  output$modal_title <- renderText({
    req(length(modal_state$ids) >= 1, modal_state$kw, modal_state$idx)
    cur_id <- modal_state$ids[modal_state$idx]
    paste0("Award ", cur_id, " — ", modal_state$kw)
  })
  
  output$abs_html <- renderUI({
    req(length(modal_state$ids) >= 1, modal_state$idx)
    cur_id <- modal_state$ids[modal_state$idx]
    abs_row <- hegs_df %>%
      dplyr::filter(.data$award_id == cur_id) %>%
      dplyr::transmute(award_id,
                       abstract = stringr::str_replace_all(abstract_clean, "-", " "))
    if (!nrow(abs_row) || is.na(abs_row$abstract[1]) || !nzchar(abs_row$abstract[1])) {
      return(HTML("<em>Abstract not found.</em>"))
    }
    HTML(highlight_phrase(abs_row$abstract[1], modal_state$kw))
  })
  
  observeEvent(input$preview_abs, {
    req(!is.null(kw_sub), !is.null(input$kw_for_preview), nrow(kw_sub) > 0)
    row_idx <- which(kw_sub$phrase == input$kw_for_preview)
    if (length(row_idx) != 1) { showNotification("Keyword not found in active set.", type = "error"); return() }
    
    ids <- kw_sub$award_ids[[row_idx]]
    ids <- ids[!is.na(ids) & nzchar(ids)]
    ids <- intersect(ids, hegs_df$award_id)
    if (!length(ids)) { showNotification("No abstracts found for this keyword.", type = "warning"); return() }
    
    modal_state$ids <- ids
    modal_state$idx <- 1
    modal_state$kw  <- input$kw_for_preview
    
    showModal(modalDialog(
      title = textOutput("modal_title"),
      easyClose = TRUE, size = "l",
      footer = tagList(
        actionButton("prev_abs", "← Prev"),
        uiOutput("pos_indicator"),
        actionButton("next_abs", "Next →"),
        modalButton("Close")
      ),
      div(style = "max-height: 60vh; overflow-y: auto;",
          htmlOutput("abs_html"))
    ))
  })
  
  output$pos_indicator <- renderUI({
    req(length(modal_state$ids) >= 1)
    span(style = "margin: 0 10px;", paste0(modal_state$idx, " of ", length(modal_state$ids)))
  })
  
  observeEvent(input$next_abs, {
    req(length(modal_state$ids) >= 1)
    n <- length(modal_state$ids)
    modal_state$idx <- (modal_state$idx %% n) + 1
  })
  observeEvent(input$prev_abs, {
    req(length(modal_state$ids) >= 1)
    n <- length(modal_state$ids)
    modal_state$idx <- ((modal_state$idx - 2) %% n) + 1
  })
}

shinyApp(ui, server)
