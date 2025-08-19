#!/usr/bin/env Rscript
# ================================================================
# File: 30_embed_with_finetuned.R
# Purpose: Use your fine-tuned SentenceTransformer to embed
#          abstracts and keywords, then write:
#            - data/interim/abstract_embeddings.parquet
#            - data/interim/keyword_embeddings.parquet
#            - data/interim/keyword_similarity.parquet (upper-tri long)
# ================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(tibble)
  library(arrow)
  library(reticulate)
})

## ---------------- Parameters ----------------
in_abstracts_csv <- "data/interim/HEGS_clean_df.csv"      # award_id, abstract_clean
in_keywords_csv  <- "data/interim/keyword_candidates.csv" # kid, phrase
output_dir       <- "data/interim/finetune"

model_dir        <- "models/sbert_final_folds1-4"         # <-- your fine-tuned model save dir
abs_batch_size   <- 128L
kw_batch_size    <- 256L
sim_block        <- 5000L    # keyword similarity block size (rows); bump if you have lots of RAM

out_abs_parquet  <- file.path(output_dir, "abstract_embeddings.parquet")
out_kw_parquet   <- file.path(output_dir, "keyword_embeddings.parquet")

# Python / Torch settings (feel free to tweak)
Sys.setenv(
  TOKENIZERS_PARALLELISM = "false",
  OMP_NUM_THREADS = "4",
  MKL_NUM_THREADS = "4"
)
torch_threads <- 1L

## ---------------- Utilities ----------------
source('R/proj_utilities.R')

## ---------------- Load inputs ----------------
if (!file.exists(in_abstracts_csv)) stop(sprintf("Missing input: %s", in_abstracts_csv))
if (!file.exists(in_keywords_csv))  stop(sprintf("Missing input: %s", in_keywords_csv))
ensure_dir(output_dir)

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

## ---------------- Python init & model ----------------
# reticulate::use_virtualenv('r-env', required = TRUE)  # uncomment if you need a specific venv
torch <- import("torch", delay_load = TRUE)
sentence_transformers <- import("sentence_transformers", delay_load = TRUE)
try({ torch$set_num_threads(as.integer(torch_threads)) }, silent = TRUE)

logi("Loading fine-tuned SentenceTransformer: %s", model_dir)
model <- sentence_transformers$SentenceTransformer(model_dir)

## ---------------- Encode abstracts ----------------
abstracts <- as_chr(df_abs$abstract_clean)
logi("Encoding abstracts with normalize_embeddings=TRUE (batch=%d)…", abs_batch_size)
abs_emb <- model$encode(
  abstracts,
  convert_to_numpy = TRUE,
  batch_size = as.integer(abs_batch_size),
  normalize_embeddings = TRUE,
  show_progress_bar = TRUE
)
if (is.null(dim(abs_emb)) || nrow(abs_emb) != length(abstracts)) {
  stop("Abstract embedding shape mismatch.")
}
D <- ncol(abs_emb)
abs_cols <- paste0("dim_", seq_len(D))
abs_tbl_df <- tibble(award_id = df_abs$award_id) %>%
  bind_cols(as_tibble(abs_emb, .name_repair = ~abs_cols)) %>%
  arrange(award_id)

if (any(is.na(abs_tbl_df$award_id))) stop("NA award_id found in abstracts.")
if (anyDuplicated(abs_tbl_df$award_id)) stop("Duplicate award_id in abstracts.")

logi("Writing: %s", out_abs_parquet)
write_parquet(abs_tbl_df, out_abs_parquet)

## ---------------- Encode keywords ----------------
kw_texts <- as_chr(df_kw$phrase)
logi("Encoding keywords with normalize_embeddings=TRUE (batch=%d)…", kw_batch_size)
kw_emb <- model$encode(
  kw_texts,
  convert_to_numpy = TRUE,
  batch_size = as.integer(kw_batch_size),
  normalize_embeddings = TRUE,
  show_progress_bar = TRUE
)
if (is.null(dim(kw_emb)) || nrow(kw_emb) != length(kw_texts)) {
  stop("Keyword embedding shape mismatch.")
}
if (ncol(kw_emb) != D) {
  logi("NOTE: abstract dim = %d, keyword dim = %d; continuing with keyword dim.", D, ncol(kw_emb))
  D <- ncol(kw_emb)
  abs_cols <- paste0("dim_", seq_len(D))
}
kw_cols <- paste0("dim_", seq_len(D))
kw_tbl_df <- tibble(kid = df_kw$kid, phrase = df_kw$phrase) %>%
  bind_cols(as_tibble(kw_emb, .name_repair = ~kw_cols)) %>%
  arrange(kid)

if (any(is.na(kw_tbl_df$kid))) stop("NA kid found in keywords.")
if (anyDuplicated(kw_tbl_df$kid)) stop("Duplicate kid in keywords.")

logi("Writing: %s", out_kw_parquet)
write_parquet(kw_tbl_df, out_kw_parquet)

logi("Done. Parquet files ready in %s", output_dir)
