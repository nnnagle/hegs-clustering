#!/usr/bin/env Rscript
# ================================================================
# File: 20_split_kw_pairs_kfold.R
# Purpose: Assign keyword-pair edges to K roughly-equal, non-overlapping
#          folds by placing whole connected components into a single fold.
#
# Input:
#   - /data/interim/kw_pairs.csv   (requires: kw1, kw2; other cols preserved)
#
# Outputs (all in /data/interim/):
#   - kw_pairs_with_folds.csv      # one row per edge, plus `fold`
#   - kw_split_components.csv      # per-component stats & fold
#   - kw_split_keywords.csv        # keyword -> fold assignment
#   - kw_split_summary.csv         # fold-level weights & counts
# ================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(igraph)
  library(tidyr)
})

## ---------------- Parameters ----------------
setwd('~/Dropbox/git/hegs-clustering/')
in_pairs_csv <- "data/interim/kw_pairs.csv"
out_dir      <- "data/interim"

k_folds    <- 5         # number of folds (>= 2)
balance_by <- "edges"   # "edges" or "nodes"
seed       <- 42        # random seed for reproducibility
verbose    <- TRUE      # print logs?

## ---------------- Derived output paths ----------------
out_pairs_csv   <- file.path(out_dir, "kw_pairs_with_folds.csv")
out_comp_csv    <- file.path(out_dir, "kw_split_components.csv")
out_kw_csv      <- file.path(out_dir, "kw_split_keywords.csv")
out_summary_csv <- file.path(out_dir, "kw_split_summary.csv")

## ---------------- Utilities ----------------
timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
logi <- function(...) if (verbose) cat(sprintf("[%s] ", timestamp()), sprintf(...), "\n")
fail <- function(msg, ...) stop(sprintf(msg, ...), call. = FALSE)

## ---------------- Validate params ----------------
if (!(balance_by %in% c("edges", "nodes"))) fail("BALANCE_BY must be 'edges' or 'nodes'.")
if (is.na(k_folds) || k_folds < 2L) fail("K_FOLDS must be >= 2 (got %s).", as.character(k_folds))
set.seed(seed)

## ---------------- Load & clean input ----------------
if (!file.exists(in_pairs_csv)) fail("Missing input: %s", in_pairs_csv)
logi("Reading keyword pairs: %s", in_pairs_csv)
df_raw <- read_csv(in_pairs_csv, show_col_types = FALSE)

req_cols <- c("kw1", "kw2")
miss <- setdiff(req_cols, names(df_raw))
if (length(miss)) fail("Input missing columns: %s", paste(miss, collapse = ", "))

df <- df_raw %>%
  mutate(kw1 = as.character(kw1), kw2 = as.character(kw2)) %>%
  filter(!is.na(kw1), !is.na(kw2), nchar(kw1) > 0, nchar(kw2) > 0)
if (nrow(df) == 0L) fail("No valid rows after cleaning.")

extra_cols <- setdiff(names(df), c("kw1", "kw2"))
df <- df %>%
  mutate(a = pmin(kw1, kw2), b = pmax(kw1, kw2)) %>%
  filter(a != b) %>%
  distinct(a, b, !!!rlang::syms(extra_cols), .keep_all = TRUE) %>%
  transmute(kw1 = a, kw2 = b, !!!rlang::syms(extra_cols))
if (nrow(df) == 0L) fail("No edges remain after removing self-loops and duplicates.")

## ---------------- Graph & components ----------------
g <- graph_from_data_frame(df[, c("kw1", "kw2")], directed = FALSE)
comps <- components(g)
comp_id <- comps$membership
nodes   <- names(comp_id)

comp_sizes_nodes <- as.data.frame(table(comp_id), stringsAsFactors = FALSE) |>
  transmute(comp_id = as.integer(comp_id), n_nodes = as.integer(Freq))

edge_dt <- df |> mutate(comp = comp_id[match(kw1, nodes)])
edges_in_comp <- edge_dt |>
  count(comp, name = "n_edges") |>
  rename(comp_id = comp)

comp_stats <- comp_sizes_nodes |>
  left_join(edges_in_comp, by = "comp_id") |>
  mutate(n_edges = coalesce(n_edges, 0L))
if (nrow(comp_stats) == 0L) fail("No components found; cannot split.")

comp_stats <- comp_stats |>
  mutate(weight = if (balance_by == "edges") n_edges else n_nodes)
if (all(comp_stats$weight == 0L)) {
  logi("All component %s are zero; falling back to node counts.", balance_by)
  comp_stats$weight <- comp_stats$n_nodes
}

comp_stats <- comp_stats |>
  arrange(desc(weight), sample.int(n()))

## ---------------- Greedy K-way partition ----------------
fold_ids <- seq_len(k_folds)
fold_weight <- setNames(rep(0L, k_folds), fold_ids)
comp_fold <- integer(nrow(comp_stats))

for (i in seq_len(nrow(comp_stats))) {
  w <- comp_stats$weight[i]
  min_w <- min(fold_weight)
  cands <- as.integer(names(fold_weight[fold_weight == min_w]))
  chosen <- if (length(cands) == 1L) cands else sample(cands, 1L)
  comp_fold[i] <- chosen
  fold_weight[as.character(chosen)] <- fold_weight[as.character(chosen)] + w
}
comp_stats$fold <- comp_fold

## ---------------- Keyword & edge assignment ----------------
kw_assign <- tibble::tibble(
  keyword = nodes,
  comp_id = comp_id,
  fold    = comp_stats$fold[match(comp_id, comp_stats$comp_id)]
) %>% arrange(keyword)

dup_kw <- kw_assign %>% count(keyword) %>% filter(n > 1)
if (nrow(dup_kw) > 0L) fail("Duplicate keyword assignment detected.")

df_with_folds <- df %>%
  left_join(select(kw_assign, keyword, fold), by = c("kw1" = "keyword")) %>%
  rename(fold1 = fold) %>%
  left_join(select(kw_assign, keyword, fold), by = c("kw2" = "keyword")) %>%
  rename(fold2 = fold)

dropped <- nrow(df_with_folds %>% filter(fold1 != fold2))
if (dropped != 0L) fail("%d edges span multiple folds.", dropped)

pairs_with_fold <- df_with_folds %>%
  transmute(kw1, kw2, !!!rlang::syms(extra_cols), fold = fold1)

## ---------------- Outputs ----------------
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(
  comp_stats %>% transmute(comp_id, n_nodes, n_edges, weight, fold) %>% arrange(desc(weight), comp_id),
  out_comp_csv
)
write_csv(kw_assign, out_kw_csv)
write_csv(pairs_with_fold, out_pairs_csv)

summary_tbl <- pairs_with_fold %>%
  count(fold, name = "n_edges") %>%
  tidyr::complete(fold = fold_ids, fill = list(n_edges = 0L)) %>%
  mutate(weight = as.integer(fold_weight[as.character(fold)])) %>%
  mutate(
    edges_share  = ifelse(sum(n_edges) > 0, n_edges / sum(n_edges), NA_real_),
    weight_share = ifelse(sum(weight) > 0, weight / sum(weight), NA_real_)
  ) %>%
  arrange(fold)
write_csv(summary_tbl, out_summary_csv)

## ---------------- Logging ----------------
logi("K-fold split complete: K=%d, balance_by=%s", k_folds, balance_by)
logi("Fold weights: %s", paste(summary_tbl$weight, collapse = ", "))
logi("Fold edge counts: %s", paste(summary_tbl$n_edges, collapse = ", "))
logi("Wrote:\n  - %s\n  - %s\n  - %s\n  - %s",
     out_pairs_csv, out_comp_csv, out_kw_csv, out_summary_csv)
logi("Done.")
