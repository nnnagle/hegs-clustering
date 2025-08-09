#!/usr/bin/env Rscript
# ================================================================
# File: 15_sentence_transformers.R
# Purpose: Generate sentence-transformer embeddings for abstracts
#          and keyword candidates, then save as Parquet (analysis-friendly).
#
# Inputs (CSV):
#   - data/interim/HEGS_clean_df.csv       (requires: award_id, abstract_clean)
#   - data/interim/keyword_candidates.csv  (requires: kid, phrase)
#
# Outputs (Parquet):
#   - data/interim/abstract_embeddings.parquet   # award_id, dim_1..dim_D
#   - data/interim/keyword_embeddings.parquet    # kid, phrase, dim_1..dim_D
#   - data/interim/keyword_similarity.parquet    # kid1, kid2, cosine (upper-tri incl diag)
# ================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(reticulate)
  library(arrow)
})

## ---------------- Parameters ----------------
in_abstracts_csv <- "data/interim/HEGS_clean_df.csv"
in_keywords_csv  <- "data/interim/keyword_candidates.csv"
output_dir       <- "data/interim"

out_abs_parquet  <- file.path(output_dir, "abstract_embeddings.parquet")
out_kw_parquet   <- file.path(output_dir, "keyword_embeddings.parquet")
out_sim_parquet  <- file.path(output_dir, "keyword_similarity.parquet")

embedding_model   <- "all-mpnet-base-v2"
encode_batch_size <- 64L   # reduce if you hit memory limits

# Python / Torch settings
Sys.setenv(
  TOKENIZERS_PARALLELISM = "false",
  OMP_NUM_THREADS = "4",
  MKL_NUM_THREADS = "4"
)
torch_threads <- 1L

## ---------------- Utilities ----------------
timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
logi <- function(...) cat(sprintf("[%s] ", timestamp()), sprintf(...), "\n")
ensure_dir <- function(path) dir.create(path, recursive = TRUE, showWarnings = FALSE)

safe_l2_normalize <- function(M) {
  norms <- sqrt(rowSums(M^2))
  norms[norms == 0] <- 1
  M / norms
}

## ---------------- Load inputs ----------------
if (!file.exists(in_abstracts_csv)) stop(sprintf("Missing input: %s", in_abstracts_csv))
if (!file.exists(in_keywords_csv))  stop(sprintf("Missing input: %s", in_keywords_csv))
ensure_dir(output_dir)

logi("Reading abstracts: %s", in_abstracts_csv)
df_abs <- read_csv(in_abstracts_csv, show_col_types = FALSE) %>%
  filter(!is.na(award_id), !is.na(abstract_clean), nchar(abstract_clean) > 0)

req_abs_cols <- c("award_id", "abstract_clean")
miss_abs <- setdiff(req_abs_cols, names(df_abs))
if (length(miss_abs)) stop(sprintf("Abstracts missing columns: %s", paste(miss_abs, collapse = ", ")))
if (nrow(df_abs) == 0L) stop("No abstracts to embed.")

logi("Reading keywords: %s", in_keywords_csv)
df_kw <- read_csv(in_keywords_csv, show_col_types = FALSE) %>%
  filter(!is.na(kid), !is.na(phrase), nchar(phrase) > 0) %>%
  mutate(kid = as.character(kid))

req_kw_cols <- c("kid", "phrase")
miss_kw <- setdiff(req_kw_cols, names(df_kw))
if (length(miss_kw)) stop(sprintf("Keywords missing columns: %s", paste(miss_kw, collapse = ", ")))
if (nrow(df_kw) == 0L) stop("No keywords to embed.")

logi("Abstracts: %d | Keywords: %d", nrow(df_abs), nrow(df_kw))

## ---------------- Python init & model ----------------
reticulate::use_virtualenv('r-env', required = TRUE) # uncomment if needed
# reticulate::use_condaenv('your-conda-env', required = TRUE)

torch <- import("torch", delay_load = TRUE)
sentence_transformers <- import("sentence_transformers", delay_load = TRUE)
try({ torch$set_num_threads(as.integer(torch_threads)) }, silent = TRUE)

logi("Loading SentenceTransformer: %s", embedding_model)
model <- sentence_transformers$SentenceTransformer(embedding_model)

## ---------------- Encode abstracts ----------------
abstracts <- as.character(df_abs$abstract_clean)
logi("Encoding abstracts (batch_size=%d)…", encode_batch_size)
abs_emb <- model$encode(
  abstracts,
  convert_to_numpy = TRUE,
  batch_size = as.integer(encode_batch_size)
)
if (is.null(dim(abs_emb)) || nrow(abs_emb) != length(abstracts)) {
  stop("Abstract embedding shape mismatch.")
}
abs_emb <- safe_l2_normalize(abs_emb)
colnames(abs_emb) <- paste0("dim_", seq_len(ncol(abs_emb)))

abs_tbl_df <- tibble::tibble(award_id = df_abs$award_id) %>%
  bind_cols(as_tibble(abs_emb)) %>%
  arrange(award_id)

if (any(is.na(abs_tbl_df$award_id))) stop("NA award_id found in abstracts.")
if (anyDuplicated(abs_tbl_df$award_id)) stop("Duplicate award_id in abstracts.")

## ---------------- Encode keywords ----------------
kw_texts <- as.character(df_kw$phrase)
logi("Encoding keywords (batch_size=%d)…", encode_batch_size)
kw_emb <- model$encode(
  kw_texts,
  convert_to_numpy = TRUE,
  batch_size = as.integer(encode_batch_size)
)
if (is.null(dim(kw_emb)) || nrow(kw_emb) != length(kw_texts)) {
  stop("Keyword embedding shape mismatch.")
}
kw_emb <- safe_l2_normalize(kw_emb)
colnames(kw_emb) <- paste0("dim_", seq_len(ncol(kw_emb)))

kw_tbl_df <- tibble::tibble(kid = df_kw$kid, phrase = df_kw$phrase) %>%
  bind_cols(as_tibble(kw_emb)) %>%
  arrange(kid)

if (any(is.na(kw_tbl_df$kid))) stop("NA kid found in keywords.")
if (anyDuplicated(kw_tbl_df$kid)) stop("Duplicate kid in keywords.")

## ---------------- Keyword–keyword cosine (upper-tri long) ----------------
logi("Computing keyword cosine similarity (%d x %d)…", nrow(kw_emb), nrow(kw_emb))
S <- kw_emb %*% t(kw_emb)  # cosines (already L2-normalized)
kid_vec <- kw_tbl_df$kid
idx <- which(upper.tri(S, diag = TRUE), arr.ind = TRUE)

kw_sim_long_df <- tibble::tibble(
  kid1   = kid_vec[idx[, 1]],
  kid2   = kid_vec[idx[, 2]],
  cosine = as.numeric(S[idx])
) %>% arrange(kid1, kid2)

if (any(is.na(kw_sim_long_df$kid1)) || any(is.na(kw_sim_long_df$kid2))) stop("NA in keyword similarity keys.")
if (any(is.na(kw_sim_long_df$cosine))) stop("NA in keyword similarity values.")

## ---------------- Write Parquet ----------------
ensure_dir(output_dir)

logi("Writing: %s", out_abs_parquet)
write_parquet(abs_tbl_df, out_abs_parquet)

logi("Writing: %s", out_kw_parquet)
write_parquet(kw_tbl_df, out_kw_parquet)

logi("Writing: %s", out_sim_parquet)
write_parquet(kw_sim_long_df, out_sim_parquet)

logi("Done. Wrote Parquet files to %s", output_dir)
