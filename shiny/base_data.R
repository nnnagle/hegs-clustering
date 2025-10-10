# shiny/base_data.R
load_base_data <- function() {
  paths <- list(
    kc_db   = "../data/interim/keyword_candidates.sqlite",
    parquet = "../data/interim/keyword_embeddings.parquet",
    hegs    = "../data/interim/HEGS_clean_df.csv"
  )
  
  # DB connect
  kc_con <- DBI::dbConnect(RSQLite::SQLite(), paths$kc_db)
  on.exit(NULL)
  onStop(function() { try(DBI::dbDisconnect(kc_con), silent = TRUE) })
  invisible(DBI::dbExecute(kc_con, "PRAGMA journal_mode=WAL;"))
  invisible(DBI::dbExecute(kc_con, "PRAGMA synchronous=NORMAL;"))
  
  # HEGS abstracts
  hegs_df <- tryCatch({
    readr::read_csv(paths$hegs, show_col_types = FALSE) %>% mutate(award_id = as.character(award_id))
  }, error = function(e) tibble(award_id = character(0), abstract = character(0)))
  
  # Candidates
  keyword_candidates <- DBI::dbGetQuery(
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
  
  # Embeddings
  kw_emb <- arrow::read_parquet(paths$parquet) %>%
    mutate(kid = as.character(kid)) %>%
    semi_join(dplyr::select(keyword_candidates, kid), by = "kid") %>%
    arrange(kid)
  dim_cols <- grep("^dim_", names(kw_emb), value = TRUE)
  if (!length(dim_cols)) stop("No embedding columns 'dim_*' found.")
  emb_mat <- as.matrix(dplyr::select(kw_emb, dplyr::all_of(dim_cols)))
  rownames(emb_mat) <- kw_emb$kid
  emb_mat <- renorm(emb_mat)
  
  # Active subset and matrices
  kids_universe <- keyword_candidates$kid %>% as.character()
  kid_sub <- keyword_candidates %>% filter(!omit) %>% pull(kid)
  if (length(kid_sub) < 2) stop("After filtering omit==TRUE, < 2 keywords remain.")
  kw_sub <- keyword_candidates %>%
    filter(kid %in% kid_sub) %>%
    arrange(match(kid, kid_sub))
  kids <- kw_sub$kid
  X <- emb_mat[kids, , drop = FALSE]
  kw_sim_full <- tcrossprod(X)
  #kw_sim_full <- emb_mat[kids,] %*% t(emb_mat[kids,])
  
  # Sparse co-occurrence
  all_docs <- sort(unique(unlist(kw_sub$award_ids)))
  doc_ids <- setNames(seq_along(all_docs), all_docs)
  kw_ids <- setNames(seq_along(kw_sub$kid), kw_sub$kid)
  i <- rep.int(x = kw_ids, times = lengths(kw_sub$award_ids))
  j <- doc_ids[unlist(kw_sub$award_ids)]
  X <- Matrix::sparseMatrix(i = i, j = j, x = 1L,
                            dims = c(nrow(kw_sub), length(all_docs)),
                            dimnames = list(kw_sub$kid, all_docs))
  C_full <- X %*% t(X); diag(C_full) <- 0; C_full <- Matrix::drop0(C_full)
  showNotification("Data Loaded", type = "message")
  
  # Reactives we can update later
  list(
    kc_con          = kc_con,
    hegs_df         = hegs_df,
    emb_mat         = emb_mat,
    kids_universe   = kids_universe,
    kw_sim_full     = kw_sim_full,
    C_full          = C_full,
    keyword_candidates = reactiveVal(keyword_candidates),
    phr_to_kid      = reactiveVal(setNames(keyword_candidates$kid, keyword_candidates$phrase)),
    kw_sub          = reactiveVal(kw_sub),
    kids            = reactiveVal(kids),
    data_ready      = reactiveVal(TRUE)
  )
}
