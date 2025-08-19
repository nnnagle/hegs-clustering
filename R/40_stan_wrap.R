# ------------------------------------------------------------------------------
# Script: 20_CETM_build_stan_inputs.R
#
# Purpose:
#   Build Stan data list for a Correlated Embedded Topic Model (CETM; keywords-only).
#   Option A (preferred): derive the doc–keyword sparse matrix from keyword_candidates.csv
#     where each row is a keyword (kid) with a comma-separated list of award_ids.
#   Option B: load a prebuilt doc–keyword matrix from disk (dense/sparse/triplets).
#   Emits sparse triplets (ii, jj, vv) plus normalized feature matrices (E, X)
#   and hyperparameters expected by the Stan program (low-rank correlations).
#
# New:
#   - Load keyword embeddings **only** from Parquet (kid, phrase, dim_1..dim_D) and PCA → prefer_M dims.
#   - Load *document embeddings* **only** from Parquet (award_id, dim_1..dim_D) at
#     data/interim/finetune/abstract_embeddings.parquet and PCA → prefer_P dims.
#   - Source proj_utilities.R for logging, dir creation, column checks, and safe character casting.
#
# Inputs (edit paths below):
#   - data/interim/keyword_candidates.csv        # preferred for counts (kid, award_ids, [omit])
#   - data/interim/finetune/keyword_embeddings.parquet    # K×D with columns: kid, phrase, dim_1..dim_D (required)
#   - data/interim/finetune/abstract_embeddings.parquet   # D×Demb with: award_id, dim_1..dim_D (required for docs)
#
# Optional (fallback if you already have counts matrix):
#   - data/interim/doc_keyword_counts.(mtx|rds|csv)  # dense/sparse/triplets; used if keyword_candidates is NULL
#
# Outputs:
#   - data/processed/stan_data_cetm_keywords.rds  (list for Stan)
#
# Algorithm:
#   1) If keyword_candidates_path exists: parse award_ids per keyword → build sparse triplets.
#      Else read counts matrix from counts_path.
#   2) Read keyword embeddings E from Parquet; extract dim_* columns,
#      compute PCA to prefer_M dims; L2-normalize rows; align rows to candidate KIDs.
#   3) Read *document embeddings* from Parquet; PCA to prefer_P dims; standardize columns; align to AWARD_IDs.
#   4) Assemble Stan data with dims, priors, and triplets; write RDS.
#
# Notes:
#   - Triplets are consolidated; Stan expects 1-based ii, jj, positive integer vv (here vv≡1 from presence).
#   - Keep E rows L2-normalized and X columns standardized for stability.
#   - Sentence-transformer outputs are unit-norm; PCA breaks norms, so we L2-normalize rows again.
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(readr)      # read_csv
  library(dplyr)      # select(where()), mutate, filter
  library(stringr)    # str_split, str_squish, str_trim
  library(purrr)      # map, map_int, map_chr
  quiet_arrow <- suppressWarnings(requireNamespace("arrow", quietly = TRUE))
})

# ---- Project utilities ----
util_path <- "R/proj_utilities.R"
if (!file.exists(util_path)) stop(sprintf("Utilities not found: %s", util_path))
source(util_path)   # provides: timestamp(), logi(), ensure_dir(), require_cols(), as_chr()

## =============================
## PARAMETERS (edit here)
## =============================

# ---- Primary input (recommended) ----
keyword_candidates_path <- "data/interim/keyword_candidates.csv"  # set to NULL to skip
use_omit_flag           <- TRUE   # drop rows with omit==TRUE when present

# ---- Cleaned abstracts (doc universe comes from here) ----
doc_clean_path          <- "data/interim/HEGS_clean_df.csv"      # must contain award_id, abstract_clean
kw_cand_path            <- "data'interim/keyword_candidates.csv"

# Keyword embeddings (Parquet ONLY)
kw_emb_path_parquet     <- "data/interim/finetune/keyword_embeddings.parquet"  # kid, phrase, dim_1..dim_D

# Document embeddings (Parquet ONLY)
doc_emb_path_parquet    <- "data/interim/finetune/abstract_embeddings.parquet" # award_id, dim_1..dim_D

# Output
out_rds                 <- "data/processed/stan_data_cetm_keywords.rds"

# ---- Model hyperparameters / dims ----
L_topics                <- 40L   # number of topics
R_rank                  <- 8L    # low-rank factor dimension (≪ L)
prefer_M                <- 200L  # target embedding dimension for E (PCA rank)
prefer_P                <- 200L  # target PCA dimension for document embeddings

beta_scale              <- 0.5   # prior sd for topic embeddings scale
W_scale                 <- 0.1   # prior sd for covariate weights
Lambda_scale            <- 0.5   # prior sd for factor loadings

## =============================
## UTILITIES (model-specific)
## =============================

# Normalize each row to unit L2 norm.
# Returns a numeric matrix with each nonzero row scaled to have ||row||_2 = 1.
l2_normalize_rows <- function(M) {
  M <- as.matrix(M)
  rs <- sqrt(rowSums(M * M))
  rs[rs == 0] <- 1
  M / rs
}

# Column-standardize a matrix to zero mean and unit variance per column.
# Returns a numeric matrix with NA-safe centering and scaling (sd=1 when constant).
standardize_cols <- function(M) {
  M <- as.matrix(M)
  mu <- colMeans(M)
  sdv <- sqrt(colSums((M - rep(1, nrow(M)) %*% t(mu))^2) / pmax(1, nrow(M) - 1))
  sdv[sdv == 0] <- 1
  sweep(sweep(M, 2, mu, "-"), 2, sdv, "/")
}

# Merge duplicate (i,j) entries in sparse triplets by summing counts.
# Returns a list(ii,jj,vv) with 1-based indices sorted by row then column.
coalesce_triplets <- function(i, j, v, D, K) {
  i <- as.integer(i); j <- as.integer(j); v <- as.integer(round(v))
  if (length(v) == 0L) return(list(ii = integer(0), jj = integer(0), vv = integer(0)))
  key <- paste(i, j, sep = "_")
  ukey <- unique(key)
  idx <- match(key, ukey)
  ii <- tapply(i, idx, `[`, 1L)
  jj <- tapply(j, idx, `[`, 1L)
  vv <- tapply(v, idx, sum)
  ord <- order(ii, jj)
  list(ii = as.integer(ii[ord]), jj = as.integer(jj[ord]), vv = as.integer(vv[ord]))
}

# Reduce dimensionality via PCA (SVD) to 'target' columns using prcomp.
# Returns the projected data matrix (observations × target PCs).
pca_reduce <- function(X, target = 200L) {
  X <- as.matrix(X)
  if (ncol(X) <= target) return(X)
  pc <- prcomp(X, center = TRUE, scale. = FALSE, rank. = target)
  pc$x
}



## =============================
## LOAD INPUTS & VALIDATE
## =============================

logi("Reading abstracts: %s", doc_clean_path)
abs_df <- read_csv(doc_clean_path, show_col_types = FALSE) %>%
  filter(!is.na(award_id), !is.na(abstract_clean), nchar(abstract_clean) > 0)

require_cols(abs_df, c("award_id", "abstract_clean"), "Abstracts")
if (nrow(abs_df) == 0L) stop("No abstracts to embed.")


logi("Reading keywords: %s", keyword_candidates_path)
kw_df <- read_csv(keyword_candidates_path, show_col_types = FALSE) %>%
  filter(!is.na(kid), !is.na(phrase), nchar(phrase) > 0) %>%
  mutate(kid = as.character(kid))
require_cols(kw_df, c("kid","phrase"), "Keywords")
if (nrow(kw_df) == 0L) stop("No keywords to embed.")

logi("Abstracts: %d | Keywords: %d", nrow(abs_df), nrow(kw_df))

logi("Reading abstract embeddings: %s", doc_emb_path_parquet)
abs_emb <- arrow::read_parquet(doc_emb_path_parquet)
require_cols(abs_emb, "award_id", sprintf("Parquet matrix (%s)", doc_emb_path_parquet))
ids <- abs_emb[['award_id']]
dim_cols <- grep("^dim_[0-9]+$", names(abs_emb), value = TRUE)
abs_mat <- as.matrix(abs_emb[,dim_cols])
dimnames(abs_mat)[[1]] <- ids


logi("Reading keyword embeddings: %s", doc_emb_path_parquet)
kw_emb <- arrow::read_parquet(kw_emb_path_parquet)
require_cols(kw_emb, c("kid","phrase"), sprintf("Parquet matrix (%s)", kw_emb_path_parquet))
ids <- kw_emb[['kid']]
dim_cols <- grep("^dim_[0-9]+$", names(kw_emb), value = TRUE)
kw_mat <- as.matrix(kw_emb[,dim_cols])
dimnames(kw_mat)[[1]] <- ids


kw_pca <- pca_reduce(kw_mat, target=prefer_M)
abs_pca <- pca_reduce(abs_mat, target=prefer_P)


logi("Creating Document-keyword matrix")
col_ids <- rep(kw_df$kid, kw_df$doc_count)
row_ids <- unlist(strsplit(kw_df$award_ids,','))
row_num <- seq(1,nrow(abs_df))
names(row_num) <- abs_df$award_id
row_ids <- row_num[row_ids]









# Early assertions
if (!file.exists(keyword_candidates_path)) {
  stop(sprintf("keyword_candidates_path not found: %s", keyword_candidates_path))
}
if (!file.exists(doc_clean_path)) {
  stop(sprintf("Cleaned abstracts CSV not found: %s", doc_clean_path))
}

# Document universe from cleaned abstracts (award_id)
docs <- load_doc_universe(doc_clean_path)

# Build counts from keyword candidates (flattened)
counts_from_kc <- build_counts_from_keyword_candidates(
  keyword_candidates_path,
  use_omit = use_omit_flag,
  docs_pre = docs
)
D <- counts_from_kc$D; K <- counts_from_kc$K
ii <- counts_from_kc$ii; jj <- counts_from_kc$jj; vv <- counts_from_kc$vv
docs <- counts_from_kc$docs; kids <- counts_from_kc$kids
logi("Counts derived from keyword_candidates: D=%d, K=%d, NNZ=%d, density=%.5f", D, K, length(vv), length(vv)/(D*K))

# 2) Keyword embeddings (Parquet ONLY; align to KIDs; PCA inside)
if (!file.exists(kw_emb_path_parquet)) stop(sprintf("Keyword embeddings Parquet not found: %s", kw_emb_path_parquet))
logi("Reading keyword embeddings (Parquet): %s", kw_emb_path_parquet)
E <- read_parquet_matrix(kw_emb_path_parquet, id_col = "kid")
orig_kw_dim <- ncol(E)
if (orig_kw_dim > prefer_M) logi("Computing PCA of keyword embeddings: %d → %d dims", orig_kw_dim, prefer_M)
E <- pca_reduce(E, target = prefer_M)
E <- l2_normalize_rows(E)  # re-normalize rows after PCA
E <- 

# Align embeddings to kids (from counts)
if (!is.null(emb_ids)) {
  idx <- match(kids, emb_ids)
  if (any(is.na(idx))) {
    missing_ids <- kids[is.na(idx)]
    stop(sprintf("Embeddings missing %d KIDs (e.g., %s)", length(missing_ids), paste(utils::head(missing_ids, 5), collapse = ", ")))
  }
  E <- E[idx, , drop = FALSE]
} else {
  if (nrow(E) != K) stop(sprintf("Embedding rows (%d) must equal K (%d) when no ID column present.", nrow(E), K))
}

M <- ncol(E)
logi("Embeddings ready: K=%d, M=%d (PCA applied=%s, rows L2-normalized, aligned)", K, M, ifelse(pca_applied_kw, "yes", "no"))

# 3) Document embeddings (Parquet ONLY) → PCA → covariates (align to award_ids)
if (!file.exists(doc_emb_path_parquet)) stop(sprintf("Document embeddings Parquet not found: %s", doc_emb_path_parquet))
logi("Reading document embeddings (Parquet): %s", doc_emb_path_parquet)
X_in <- read_parquet_matrix(doc_emb_path_parquet, id_col = "award_id")
orig_doc_dim <- ncol(X_in$mat)
if (orig_doc_dim > prefer_P) logi("Computing PCA of document embeddings: %d → %d dims", orig_doc_dim, prefer_P)
X <- pca_reduce(X_in$mat, target = prefer_P)
# Standardize columns post-PCA (keeps scale comparable to prior assumptions)
X <- standardize_cols(X)
doc_ids <- X_in$ids
pca_applied_doc <- orig_doc_dim > prefer_P

# Align X to document order (docs)
if (!is.null(doc_ids)) {
  idx <- match(docs, doc_ids)
  if (any(is.na(idx))) {
    missing_docs <- docs[is.na(idx)]
    stop(sprintf("Doc covariates missing %d award_ids (e.g., %s)", length(missing_docs), paste(utils::head(missing_docs, 5), collapse = ", ")))
  }
  X <- X[idx, , drop = FALSE]
} else {
  if (nrow(X) != D) stop(sprintf("Doc covariate rows (%d) must equal D (%d) when no ID column present.", nrow(X), D))
}

P <- ncol(X)
logi("Doc covariates ready: D=%d, P=%d (PCA applied=%s, columns standardized, aligned)", D, P, ifelse(pca_applied_doc, "yes", "no"))

## =============================
## FINAL CONSOLIDATION & CHECKS
## =============================

# Coalesce duplicated (i,j) just in case (shouldn't happen with presence data)
trip2 <- coalesce_triplets(ii, jj, vv, D, K)
ii <- trip2$ii; jj <- trip2$jj; vv <- trip2$vv
NNZ <- length(vv)

# Basic checks
if (any(vv <= 0L)) stop("All counts (vv) must be positive integers.")
if (length(ii) != length(jj) || length(ii) != length(vv)) stop("Triplet length mismatch.")
if (any(ii < 1L | ii > D)) stop("ii out of bounds.")
if (any(jj < 1L | jj > K)) stop("jj out of bounds.")

## =============================
## ASSEMBLE & WRITE STAN DATA
## =============================

stan_data <- list(
  D = as.integer(D), K = as.integer(K), L = as.integer(L_topics),
  M = as.integer(M), P = as.integer(P), R = as.integer(R_rank),
  NNZ = as.integer(NNZ),
  ii = as.integer(ii), jj = as.integer(jj), vv = as.integer(vv),
  E = E, X = X,
  beta_scale = beta_scale, W_scale = W_scale, Lambda_scale = Lambda_scale
)

ensure_dir(dirname(out_rds))
saveRDS(stan_data, out_rds)
logi("Saved Stan data to %s", out_rds)

logi("Done.")
