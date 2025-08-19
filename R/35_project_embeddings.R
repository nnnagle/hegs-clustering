#!/usr/bin/env Rscript
# ================================================================
# File: 35_project_embeddings.R
# Purpose: Project embeddings to lower dimension


## ---------------- Utilities ----------------
source('R/proj_utilities.R')

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tibble)
  library(arrow)
})

## ---------------- Parameters ----------------
in_abstracts_csv <- "data/interim/HEGS_clean_df.csv"      # award_id, abstract_clean
in_keywords_csv  <- "data/interim/keyword_candidates.csv" # kid, phrase

embeddings_dir       <- "data/interim/finetune"
abs_parquet  <- file.path(embeddings_dir, "abstract_embeddings_temp.parquet")
kw_parquet   <- file.path(embeddings_dir, "keyword_embeddings_temp.parquet")


## ---------------- Load inputs ----------------
if (!file.exists(in_abstracts_csv)) stop(sprintf("Missing input: %s", in_abstracts_csv))
if (!file.exists(in_keywords_csv))  stop(sprintf("Missing input: %s", in_keywords_csv))
ensure_dir(embeddings_dir)

logi("Reading abstracts: %s", in_abstracts_csv)
df_abs <- read_csv(in_abstracts_csv, show_col_types = FALSE) %>%
  filter(!is.na(award_id), !is.na(abstract_clean), nchar(abstract_clean) > 0)

require_cols(df_abs, c("award_id", "abstract_clean"), "Abstracts")
if (nrow(df_abs) == 0L) stop("No abstracts to embed.")

logi("Reading keywords: %s", in_keywords_csv)
df_kw <- read_csv(in_keywords_csv, show_col_types = FALSE) %>%
  filter(!is.na(kid), !is.na(phrase), nchar(phrase) > 0) %>%
  mutate(kid = as.character(kid))
require_cols(df_kw, c("kid", "phrase"), "Keywords")
if (nrow(df_kw) == 0L) stop("No keywords to embed.")

logi("Abstracts: %d | Keywords: %d", nrow(df_abs), nrow(df_kw))

# Generic loader: returns a list with matrix X and id/meta
read_embeddings_matrix <- function(path, id_col, text_col = NULL) {
  df <- read_parquet(path)
  if (!id_col %in% names(df)) stop(sprintf("Column '%s' not found in %s", id_col, path))
  
  # pick and numerically sort dim_* columns
  dim_cols <- grep("^dim_\\d+$", names(df), value = TRUE)
  if (!length(dim_cols)) stop("No 'dim_*' columns found.")
  order_idx <- order(as.integer(sub("^dim_", "", dim_cols)))
  dim_cols <- dim_cols[order_idx]
  
  # build numeric matrix
  X <- as.matrix(df[, dim_cols, drop = FALSE])
  storage.mode(X) <- "double"
  row_ids <- as.character(df[[id_col]])
  rownames(X) <- row_ids
  
  meta_cols <- c(id_col, text_col)
  meta_cols <- meta_cols[meta_cols %in% names(df)]
  meta <- if (length(meta_cols)) df[, meta_cols, drop = FALSE] else NULL
  
  list(X = X, ids = row_ids, dims = ncol(X), meta = meta)
}

# Abstracts: award_id + dim_* -> matrix with rownames = award_id
abs_emb <- read_embeddings_matrix(
  path   = abs_parquet,
  id_col = "award_id"
)
# access matrix:
abs_mat <- abs_emb$X

# Keywords: kid, phrase + dim_* -> matrix with rownames = kid
kw_emb <- read_embeddings_matrix(
  path     = kw_parquet,
  id_col   = "kid",
  text_col = "phrase"   # kept in kw_emb$meta if you want it
)
kw_mat <- kw_emb$X[df_kw$kid[!df_kw$omit],]

logi("Abstract embedding: %d x %d  |  Keyword embedding: %d x %d",
     nrow(abs_mat), ncol(abs_mat), nrow(kw_mat), ncol(kw_mat))

# assumes: kw_mat (rows=keywords, cols=dim_1..dim_D), kw_emb$meta$phrase, and logi()

library(FNN)

# --- PCA (center only; embeddings are already roughly unit-norm) ---
logi("Running prcomp on keyword embeddings…")
pca_kw <- prcomp(kw_mat, center = TRUE, scale. = FALSE)
eig <- pca_kw$sdev^2
evr <- eig / sum(eig)
logi("Explained variance ratio PC1–PC5: %s", paste(round(evr[1:5], 4), collapse = ", "))

# --- PC1 'genericity' sanity check (one-time info) ---
X  <- kw_mat
Xc <- scale(X, center = TRUE, scale = FALSE)
mu <- colMeans(X); mu <- mu / sqrt(sum(mu^2))
pc1_scores   <- as.numeric(Xc %*% pca_kw$rotation[,1])
centroid_sim <- as.numeric(X %*% mu)
rho_centroid <- cor(pc1_scores, centroid_sim, method = "spearman")
logi("corr(PC1 scores, similarity-to-centroid) = %.3f  (high ⇒ genericity axis)", rho_centroid)

# --- helpers ---
renorm <- function(M){ r <- sqrt(rowSums(M^2)); r[r == 0] <- 1; sweep(M, 1, r, "/") }
one_nn_counts <- function(A){
  idx <- FNN::get.knnx(A, A, k = 2)$nn.index[, 2]   # nearest neighbor excluding self
  tabulate(idx, nbins = nrow(A))
}
jaccard <- function(a, b) length(intersect(a, b)) / length(unique(c(a, b)))

# --- baseline neighbors for overlap (r=0) ---
K_JACCARD <- 10L
X_base    <- renorm(X)
base_knn  <- FNN::get.knn(X_base, k = K_JACCARD + 1)$nn.index[, -1, drop = FALSE]

# --- iterate r = 0..4 and report metrics ---
R_MAX <- 4L
V     <- pca_kw$rotation
results <- vector("list", R_MAX + 1L)

for (r in 0:R_MAX) {
  if (r == 0) {
    Xr <- X_base
  } else {
    Vr     <- V[, 1:r, drop = FALSE]              # D x r
    scores <- Xc %*% Vr                            # N x r
    Xc_r   <- Xc - scores %*% t(Vr)               # remove top-r PCs
    Xr     <- renorm(Xc_r)
  }
  
  # hubness (1-NN counts)
  h    <- one_nn_counts(Xr)
  hub_mean <- mean(h)
  hub_p95  <- unname(quantile(h, 0.95))
  hub_max  <- max(h)
  
  # neighbor overlap vs baseline at k=10
  knn_r <- FNN::get.knn(Xr, k = K_JACCARD + 1)$nn.index[, -1, drop = FALSE]
  overlap <- vapply(seq_len(nrow(knn_r)), function(i) jaccard(base_knn[i, ], knn_r[i, ]), numeric(1))
  j_mean  <- mean(overlap)
  j_median <- median(overlap)
  
  logi("r=%d  hub_mean=%.2f  p95=%d  max=%d  |  Jaccard(k=%d) vs r=0: mean=%.3f median=%.3f",
       r, hub_mean, hub_p95, hub_max, K_JACCARD, j_mean, j_median)
  
  results[[r + 1L]] <- data.frame(
    r = r,
    hub_mean = hub_mean,
    hub_p95  = hub_p95,
    hub_max  = hub_max,
    jaccard_mean   = j_mean,
    jaccard_median = j_median,
    stringsAsFactors = FALSE
  )
}

summary_df <- do.call(rbind, results)
logi("Summary table (r = PCs removed):")
print(summary_df)

# ================== ABSTRACT EMBEDDINGS ANALYSIS ==================

library(FNN)

# Optional sampling cap for kNN speed on large corpora
SAMPLE_ABS  <- 10000L   # set 0 to use all rows
K_JACCARD   <- 10L      # neighbor size for overlap metric
R_MAX       <- 4L       # remove 0..R_MAX PCs

# helpers (redeclare safely)
renorm <- function(M){ r <- sqrt(rowSums(M^2)); r[r == 0] <- 1; sweep(M, 1, r, "/") }
one_nn_counts <- function(A){
  idx <- FNN::get.knnx(A, A, k = 2)$nn.index[, 2]; tabulate(idx, nbins = nrow(A))
}
jaccard <- function(a, b) length(intersect(a, b)) / length(unique(c(a, b)))

# --- PCA on abstracts (center only; no scaling) ---
logi("Running prcomp on abstract embeddings…")
pca_abs <- prcomp(abs_mat, center = TRUE, scale. = FALSE)
eig_a   <- pca_abs$sdev^2
evr_a   <- eig_a / sum(eig_a)
logi("Explained variance ratio (abstracts) PC1–PC5: %s", paste(round(evr_a[1:5], 4), collapse = ", "))

# --- PC1 'genericity' & length sanity checks ---
Xa  <- abs_mat
Xac <- scale(Xa, center = TRUE, scale = FALSE)
mu_a <- colMeans(Xa); mu_a <- mu_a / sqrt(sum(mu_a^2))
pc1_scores_a   <- as.numeric(Xac %*% pca_abs$rotation[, 1])
centroid_sim_a <- as.numeric(Xa %*% mu_a)
rho_centroid_a <- cor(pc1_scores_a, centroid_sim_a, method = "spearman")
logi("Abstracts: corr(PC1 scores, similarity-to-centroid) = %.3f", rho_centroid_a)

# correlate with abstract length (characters & tokens), if available
if (all(c("award_id","abstract_clean") %in% names(df_abs))) {
  abs_text_map <- setNames(df_abs$abstract_clean, df_abs$award_id)
  abs_text <- abs_text_map[rownames(abs_mat)]
  n_chars_a  <- nchar(abs_text)
  n_tokens_a <- stringr::str_count(abs_text, "\\S+")
  rho_len  <- suppressWarnings(cor(pc1_scores_a, n_tokens_a, method = "spearman", use = "complete.obs"))
  rho_char <- suppressWarnings(cor(pc1_scores_a, n_chars_a,  method = "spearman", use = "complete.obs"))
  logi("Abstracts: Spearman corr(PC1, n_tokens)=%.3f | corr(PC1, n_chars)=%.3f", rho_len, rho_char)
}

# --- Sampling for kNN if needed ---
Na <- nrow(Xa)
idx_a <- if (SAMPLE_ABS > 0L && Na > SAMPLE_ABS) sort(sample.int(Na, SAMPLE_ABS)) else seq_len(Na)
Xa_s  <- Xa[idx_a, , drop = FALSE]
Xac_s <- scale(Xa_s, center = TRUE, scale = FALSE)  # use same centering convention

# baseline neighbors for overlap (r=0)
X_base_a   <- renorm(Xa_s)
base_knn_a <- FNN::get.knn(X_base_a, k = K_JACCARD + 1)$nn.index[, -1, drop = FALSE]

# --- iterate r = 0..4 and report metrics ---
Va <- pca_abs$rotation
results_abs <- vector("list", R_MAX + 1L)

for (r in 0:R_MAX) {
  if (r == 0) {
    Xr <- X_base_a
  } else {
    Vr     <- Va[, 1:r, drop = FALSE]           # D x r
    scores <- Xac_s %*% Vr                       # S x r
    Xc_r   <- Xac_s - scores %*% t(Vr)          # remove top-r PCs
    Xr     <- renorm(Xc_r)
  }
  
  # hubness (1-NN counts)
  h <- one_nn_counts(Xr)
  hub_mean <- mean(h)
  hub_p95  <- unname(quantile(h, 0.95))
  hub_max  <- max(h)
  
  # neighbor overlap vs baseline at k=10
  knn_r <- FNN::get.knn(Xr, k = K_JACCARD + 1)$nn.index[, -1, drop = FALSE]
  overlap <- vapply(seq_len(nrow(knn_r)), function(i) jaccard(base_knn_a[i, ], knn_r[i, ]), numeric(1))
  j_mean   <- mean(overlap)
  j_median <- median(overlap)
  
  logi("ABSTRACTS r=%d  hub_mean=%.2f  p95=%d  max=%d  |  Jaccard(k=%d) vs r=0: mean=%.3f median=%.3f",
       r, hub_mean, hub_p95, hub_max, K_JACCARD, j_mean, j_median)
  
  results_abs[[r + 1L]] <- data.frame(
    r = r,
    hub_mean = hub_mean,
    hub_p95  = hub_p95,
    hub_max  = hub_max,
    jaccard_mean   = j_mean,
    jaccard_median = j_median,
    stringsAsFactors = FALSE
  )
}

summary_abs <- do.call(rbind, results_abs)
logi("ABSTRACTS summary (r = PCs removed; sample size used = %d of %d):", length(idx_a), Na)
print(summary_abs)

###############################
# install.packages(c("FNN","dplyr")) if needed
library(FNN)
library(dplyr)

# ---------- helpers ----------
renorm <- function(M){ r <- sqrt(rowSums(M^2)); r[r == 0] <- 1; sweep(M, 1, r, "/") }

# Knee (elbow) on cumulative EVR by max distance from chord
knee_index <- function(cum_evr){
  x1 <- 1; y1 <- cum_evr[1]
  x2 <- length(cum_evr); y2 <- cum_evr[length(cum_evr)]
  xs <- seq_along(cum_evr); ys <- cum_evr
  # distance point->line
  denom <- sqrt((y2 - y1)^2 + (x2 - x1)^2)
  d <- abs((y2 - y1)*xs - (x2 - x1)*ys + x2*y1 - y2*x1) / denom
  which.max(d)
}

pca_keep_sweep <- function(X,
                           k_nn = 10L,
                           sample_n = 5000L,
                           m_grid = NULL,          # if NULL, builds a sensible grid from dims
                           verbose = TRUE) {
  N <- nrow(X); D <- ncol(X)
  if (is.null(m_grid)) {
    base <- unique(pmin(D, c(8, 12, 16, 20, 24, 32, 40, 50, 64, 80, 96, 128, 160, 200, 256, 320, 384, 512, D)))
    m_grid <- sort(base)
  }
  
  # PCA (center only)
  if (verbose) logi("PCA: N=%d, D=%d", N, D)
  Xc <- scale(X, center = TRUE, scale = FALSE)
  pca <- prcomp(Xc, center = FALSE, scale. = FALSE)
  
  eig <- pca$sdev^2
  evr <- eig / sum(eig)
  cum_evr <- cumsum(evr)
  
  # Variance targets
  m90 <- which(cum_evr >= 0.90)[1]
  m95 <- which(cum_evr >= 0.95)[1]
  m99 <- which(cum_evr >= 0.99)[1]
  
  # MP bulk edge (robust sigma^2 from median eigenvalue)
  q <- D / (N - 1)
  sigma2_hat <- median(eig)
  lambda_max_mp <- sigma2_hat * (1 + sqrt(q))^2
  n_signal <- sum(eig > lambda_max_mp)
  
  # Knee
  m_knee <- knee_index(cum_evr)
  
  # Neighborhood preservation curve
  idx <- if (sample_n > 0L && N > sample_n) sort(sample.int(N, sample_n)) else seq_len(N)
  Xs  <- X[idx, , drop = FALSE]
  Xs0 <- renorm(Xs)
  base_knn <- FNN::get.knn(Xs0, k = k_nn + 1)$nn.index[, -1, drop = FALSE]
  
  V <- pca$rotation
  Xcs <- Xc[idx, , drop = FALSE]
  j_rows <- vector("list", length(m_grid))
  for (ii in seq_along(m_grid)) {
    m <- m_grid[ii]
    Sco <- Xcs %*% V[, 1:m, drop = FALSE]        # scores
    Sco <- renorm(Sco)                            # unit length for cosine~euclid
    knn_m <- FNN::get.knn(Sco, k = k_nn + 1)$nn.index[, -1, drop = FALSE]
    # Jaccard vs baseline
    jac <- vapply(seq_len(nrow(base_knn)),
                  function(i) {
                    a <- base_knn[i, ]; b <- knn_m[i, ]
                    length(intersect(a, b)) / length(unique(c(a, b)))
                  }, numeric(1))
    j_rows[[ii]] <- data.frame(m = m,
                               jaccard_mean = mean(jac),
                               jaccard_median = median(jac),
                               stringsAsFactors = FALSE)
  }
  j_tbl <- bind_rows(j_rows)
  
  # Smallest m hitting Jaccard targets (neighborhood preservation)
  m_j90 <- j_tbl %>% filter(jaccard_mean >= 0.90) %>% summarise(min_m = ifelse(n() == 0, NA_integer_, min(m))) %>% pull(min_m)
  m_j80 <- j_tbl %>% filter(jaccard_mean >= 0.80) %>% summarise(min_m = ifelse(n() == 0, NA_integer_, min(m))) %>% pull(min_m)
  
  if (verbose) {
    logi("EVR targets: 90%%→m=%s, 95%%→m=%s, 99%%→m=%s",
         ifelse(is.na(m90), "NA", m90),
         ifelse(is.na(m95), "NA", m95),
         ifelse(is.na(m99), "NA", m99))
    logi("MP bulk edge: λ_max≈%.4f  |  #eig>λ_max = %d (signal PCs)",
         lambda_max_mp, n_signal)
    logi("Knee (elbow) m≈%d", m_knee)
    logi("k-NN (k=%d) Jaccard mean by m (first few): %s",
         k_nn,
         paste(sprintf("m=%d: %.3f", head(j_tbl$m, 6), head(j_tbl$jaccard_mean, 6)), collapse=", "))
    if (!is.na(m_j90)) logi("Smallest m with Jaccard≥0.90: m=%d", m_j90) else logi("No m reached Jaccard≥0.90")
    if (!is.na(m_j80)) logi("Smallest m with Jaccard≥0.80: m=%d", m_j80) else logi("No m reached Jaccard≥0.80")
  }
  
  list(
    pca = pca,
    eigenvalues = eig,
    evr = evr,
    cum_evr = cum_evr,
    m_evr_90 = m90, m_evr_95 = m95, m_evr_99 = m99,
    lambda_max_mp = lambda_max_mp,
    n_signal = n_signal,
    m_knee = m_knee,
    jaccard_table = j_tbl,
    m_jaccard_90 = m_j90,
    m_jaccard_80 = m_j80
  )
}

######
# Keywords
logi("=== Choosing m for KEYWORDS ===")
kw_res <- pca_keep_sweep(kw_mat, k_nn = 10, sample_n = 5000)
# A blunt, defensible pick (feel free to override):
m_kw <- max(na.omit(c(kw_res$n_signal, kw_res$m_knee, kw_res$m_jaccard_90)))
logi("Chosen m for KEYWORDS (max of signal/knee/Jaccard90): %s", ifelse(length(m_kw)==0, "NA", m_kw))

# Abstracts
logi("=== Choosing m for ABSTRACTS ===")
abs_res <- pca_keep_sweep(abs_mat, k_nn = 10, sample_n = 5000)
m_abs <- max(na.omit(c(abs_res$n_signal, abs_res$m_knee, abs_res$m_jaccard_90)))
logi("Chosen m for ABSTRACTS (max of signal/knee/Jaccard90): %s", ifelse(length(m_abs)==0, "NA", m_abs))

