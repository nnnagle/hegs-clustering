# ------------------------------------------------------------------------------
# Script: 05_HEGS_clean.R
#
# Purpose:
#   Clean NSF HEGS award abstracts, remove boilerplate statutory text,
#   and deduplicate near-duplicate abstracts using sentence-transformer
#   embeddings + cosine similarity + connected components.
#
# Inputs (CSV):
#   data/raw/HEGS_awards.csv
#     - Must contain columns 'award_id' 'abstract'
#
# Output (CSV):
#   data/intermediate/HEGS_clean_df.csv
#     - Original columns plus 'abstract_clean'
#     - Deduplicated rows (one representative per near-duplicate cluster)
#
# Algorithm (high level):
#   1) Load CSV; strip statutory boilerplate and normalize the text.
#.  Deduplicate:
#   2) Encode abstracts with Sentence Transformers (all-mpnet-base-v2).
#   3) L2-normalize embeddings, compute cosine similarity (dot product).
#   4) Build graph of pairs > threshold; take one per connected component.
#   5) Write cleaned, deduplicated CSV.
#
# Notes:
#   - If you have many thousands of abstracts, a full similarity matrix
#     is O(N^2) memory/time. Consider blocking/LSH for very large N.
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(readr)      
  library(dplyr)      
  library(stringr)    
  library(reticulate) 
  library(textclean)  # replace_non_ascii
  library(igraph)     # graph-based dedup
})

## =============================
## PARAMETERS (edit here)
## =============================

# Paths
input_file   <- "data/raw/HEGS_awards.csv"
output_file  <- "data/interim/HEGS_clean_df.csv"

# Sentences to strip from abstracts (escaped parens kept for regex safety)
statutory_text <- c(
  "This award reflects NSF's statutory mission and has been deemed worthy of support through evaluation using the Foundation's intellectual merit and broader impacts review criteria.",
  "This award is funded under the American Recovery and Reinvestment Act of 2009 \\(Public Law 111-5\\)."
)

# Embedding model + encoding settings
embedding_model      <- "all-mpnet-base-v2"
encode_batch_size    <- 64L        # lower if you hit RAM limits
max_seq_length_chars <- NA_integer_ # set e.g. 2000 to truncate long docs (character-level)

# Dedup threshold (cosine)
similarity_threshold <- 0.985

# Python/Torch env knobs
Sys.setenv(
  TOKENIZERS_PARALLELISM = "false",  # quieter tokenizers
  OMP_NUM_THREADS = "4",             # adjust for your CPU
  MKL_NUM_THREADS = "4"
)
torch_threads <- 1L                  # internal torch CPU threads

## =============================
## UTILITIES
## =============================

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")

logi <- function(...) {
  cat(sprintf("[%s] ", timestamp()), sprintf(...), "\n")
}

ensure_dir <- function(path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
}

fix_hyphenation <- function(x) {
  x |>
    str_replace_all("\u00AD", "") |>  # remove soft hyphen chars if present
    # join words split by a hyphen at a line break (letters on both sides)
    str_replace_all("(?<=\\p{L})-\\s*\\r?\\n\\s*(?=\\p{L})", "") |>
    # if your line breaks were turned into long runs of spaces, also fix those:
    str_replace_all("(?<=\\p{L})-\\s{2,}(?=\\p{L})", "") |>
    # flatten remaining newlines to spaces and squish
    str_replace_all("[\\r\\n]+", " ") |>
    str_squish()
}

## =============================
## LOAD & VALIDATE INPUT
## =============================

if (!file.exists(input_file)) {
  stop(sprintf("Input file not found: %s", input_file))
}
ensure_dir(output_file)

logi("Reading: %s", input_file)
raw_df <- readr::read_csv(file = input_file, show_col_types = FALSE)

if (!"award_id" %in% names(raw_df)) {
  stop("Input must contain a column named 'award_id'.")
}

if (!"abstract" %in% names(raw_df)) {
  stop("Input must contain a column named 'abstract'.")
}

n0 <- nrow(raw_df)
logi("Loaded %d rows.", n0)

## =============================
## CLEAN TEXT
## =============================

logi("Cleaning abstracts (remove statutory, normalize unicode/whitespace).")
df <- raw_df %>%
  filter(!is.na(abstract)) %>%
  mutate(
    abstract_clean = abstract %>%
      # remove statutory boilerplate
      str_replace(statutory_text[1], " ") %>%
      str_replace(statutory_text[2], " ") %>%
      fix_hyphenation() %>%
      # pad parens and collapse newlines
      str_replace_all("\\)", " ) ") %>%
      str_replace_all("\\(", " ( ") %>%
      str_replace_all("[\\r\\n]+", " ") %>%
      # clean unicode and squeeze spaces
      textclean::replace_non_ascii() %>%
      str_squish()
  )

# Optional hard truncation (rarely needed for abstracts)
if (!is.na(max_seq_length_chars) && is.finite(max_seq_length_chars)) {
  n_truncated <- sum(nchar(df$abstract_clean) > max_seq_length_chars, na.rm = TRUE)
  
  if (n_truncated > 0) {
    logi("Hard truncating %d abstracts to max %d characters (%.1f%% of dataset).",
         n_truncated, max_seq_length_chars, 100 * n_truncated / nrow(df))
  } else {
    logi("No abstracts exceeded %d characters. No truncation applied.",
         max_seq_length_chars)
  }
  
  df <- df %>% mutate(
    abstract_clean = if_else(
      nchar(abstract_clean) > max_seq_length_chars,
      str_sub(abstract_clean, 1L, max_seq_length_chars),
      abstract_clean
    )
  )
}

n_after_na <- nrow(df)
if (n_after_na == 0L) {
  stop("No non-NA abstracts after filtering.")
}
logi("Kept %d rows with non-NA abstracts.", n_after_na)

## =============================
## PYTHON / EMBEDDINGS
## =============================

# Initialize Python env (adjust to your setup if needed)
# If you use a specific virtualenv/conda env, uncomment and edit:
reticulate::use_virtualenv('r-env', required = TRUE)
# reticulate::use_condaenv('your-conda-env', required = TRUE)

torch <- reticulate::import("torch", delay_load = TRUE)
sentence_transformers <- reticulate::import("sentence_transformers", delay_load = TRUE)

# Limit torch CPU threads (helps stability)
try({
  torch$set_num_threads(as.integer(torch_threads))
}, silent = TRUE)

logi("Loading SentenceTransformer model: %s", embedding_model)
model <- sentence_transformers$SentenceTransformer(embedding_model)

abstracts_chr <- as.character(df$abstract_clean)

logi("Encoding %d abstracts (batch_size=%d).", length(abstracts_chr), encode_batch_size)
embeddings <- model$encode(
  abstracts_chr,
  convert_to_numpy = TRUE,
  batch_size = as.integer(encode_batch_size)
)

# Defensive checks
if (is.null(dim(embeddings)) || nrow(embeddings) != length(abstracts_chr)) {
  stop("Embedding shape mismatch; check sentence-transformers output.")
}

# Normalize to unit length (avoid divide-by-zero)
row_norms <- sqrt(rowSums(embeddings^2))
row_norms[row_norms == 0] <- 1
norm_embeddings <- embeddings / row_norms

## =============================
## SIMILARITY & DEDUP
## =============================

N <- nrow(norm_embeddings)
logi("Computing cosine similarity matrix (%d x %d).", N, N)

# NOTE: This is O(N^2) memory/time. For very large N, refactor.
sim_matrix <- norm_embeddings %*% t(norm_embeddings)

if (!is.finite(similarity_threshold) || similarity_threshold <= -1 || similarity_threshold >= 1) {
  stop("similarity_threshold must be in (-1, 1); typical values are 0.95–0.995 for dedup.")
}

logi("Finding pairs with similarity > %.3f.", similarity_threshold)
similar_pairs <- which(sim_matrix > similarity_threshold, arr.ind = TRUE)

# remove self-pairs and mirror duplicates (keep i<j)
if (nrow(similar_pairs) > 0L) {
  similar_pairs <- similar_pairs[similar_pairs[, 1] < similar_pairs[, 2], , drop = FALSE]
}

n_pairs <- nrow(similar_pairs)
logi("Found %d near-duplicate pairs above threshold.", n_pairs)

# If none, no dedup — save cleaned CSV and exit
if (n_pairs == 0L) {
  logi("No near-duplicates found. Writing cleaned CSV.")
  write_csv(df, output_file)
  logi("Wrote: %s", output_file)
  quit(save = "no")
}

# Build graph and pick one representative per component
edges <- as.data.frame(similar_pairs[, 1:2, drop = FALSE])
names(edges) <- c("from", "to")

g <- graph_from_data_frame(edges, directed = FALSE)
clusters <- components(g)

# Representative: lowest index per component
deduped_in_graph <- tapply(as.integer(V(g)$name), clusters$membership, min)

all_docs   <- seq_len(N)
graph_docs <- as.integer(V(g)$name)
isolated   <- setdiff(all_docs, graph_docs)

deduplicated_idx <- sort(c(deduped_in_graph, isolated))
n_final <- length(deduplicated_idx)

logi("Deduplicated from %d to %d documents (%.1f%% retained).",
     N, n_final, 100 * n_final / N)

df_out <- df[deduplicated_idx, , drop = FALSE] %>%
  select(award_id, abstract, abstract_clean)

## =============================
## WRITE OUTPUT
## =============================

write_csv(df_out, output_file)
logi("Wrote deduplicated CSV: %s", output_file)
