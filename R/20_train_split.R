#!/usr/bin/env Rscript
# ================================================================
# File: 20_split_kw_pairs_kfold.R
# Purpose: Community-then-pack splitting for leakage-free folds.
#          1) Build graph from keyword pairs.
#          2) Detect communities in large components (Louvain).
#          3) Treat each community (or small component) as an atomic block.
#          4) Bin-pack blocks into K folds to balance weight and, optionally,
#             the similarity score distribution.
#
# Input:
#   - data/interim/kw_pairs.csv   (requires: kw1, kw2; other cols preserved)
#
# Outputs (all in data/interim/):
#   - kw_pairs_with_folds.csv      # one row per kept edge, plus `fold`
#   - kw_split_components.csv      # per-block stats & fold (provenance columns)
#   - kw_split_keywords.csv        # keyword -> fold assignment
#   - kw_split_summary.csv         # fold-level weights & counts
# ================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(igraph)
  library(tidyr)
  library(rlang)
})

## ---------------- Parameters ----------------
setwd('~/Dropbox/git/hegs-clustering/')
in_pairs_csv <- "data/interim/kw_pairs.csv"
out_dir      <- "data/interim"

k_folds      <- 5          # number of folds (>= 2)
balance_by   <- "edges"    # "edges" or "nodes"
score_col    <- NULL       # e.g., "similarity" or "score"; NULL to disable
n_score_bins <- 10         # bins for score balancing (used if score_col present)
lambda_bins  <- 1.0        # weight of score-balance term in packing objective

# Heuristics for when to split a component into micro-communities
giant_min_nodes <- 50     # split components with >= this many nodes
giant_min_edges <- 100     # OR with >= this many edges

seed       <- 42           # random seed for reproducibility
verbose    <- TRUE         # print logs?

## ---------------- Derived output paths ----------------
out_pairs_csv   <- file.path(out_dir, "kw_pairs_with_folds.csv")
out_comp_csv    <- file.path(out_dir, "kw_split_components.csv")
out_kw_csv      <- file.path(out_dir, "kw_split_keywords.csv")
out_summary_csv <- file.path(out_dir, "kw_split_summary.csv")

## ---------------- Utilities ----------------
timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
logi <- function(...) if (verbose) cat(sprintf("[%s] ", timestamp()), sprintf(...), "\n")
fail <- function(msg, ...) stop(sprintf(msg, ...), call. = FALSE)
nz <- function(x) ifelse(is.finite(x) & x != 0, x, 1)

## ---------------- Validate params ----------------
if (!(balance_by %in% c("edges", "nodes"))) fail("BALANCE_BY must be 'edges' or 'nodes'.")
if (is.na(k_folds) || k_folds < 2L) fail("K_FOLDS must be >= 2 (got %s).", as.character(k_folds))
if (!is.null(score_col) && !is.character(score_col)) fail("SCORE_COL must be a character or NULL.")
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

# Canonicalize undirected edges, remove self-loops and duplicates
df <- df %>%
  mutate(a = pmin(kw1, kw2), b = pmax(kw1, kw2)) %>%
  filter(a != b) %>%
  distinct(a, b, !!!syms(extra_cols), .keep_all = TRUE) %>%
  transmute(kw1 = a, kw2 = b, !!!syms(extra_cols))

if (nrow(df) == 0L) fail("No edges remain after removing self-loops and duplicates.")

# Validate score column if requested
if (!is.null(score_col)) {
  if (!(score_col %in% names(df))) fail("Requested score_col '%s' not found.", score_col)
  if (!is.numeric(df[[score_col]])) fail("score_col '%s' must be numeric.", score_col)
}

## ---------------- Graph & components ----------------
g <- graph_from_data_frame(df[, c("kw1", "kw2")], directed = FALSE)
comps <- components(g)
comp_id <- comps$membership
nodes   <- names(comp_id)

# Per-component sizes
comp_sizes_nodes <- as.data.frame(table(comp_id), stringsAsFactors = FALSE) |>
  transmute(orig_comp_id = as.integer(comp_id), n_nodes = as.integer(Freq))

edge_dt <- df |> mutate(orig_comp_id = comp_id[match(kw1, nodes)])
edges_in_comp <- edge_dt |>
  count(orig_comp_id, name = "n_edges")

comp_sizes <- comp_sizes_nodes |>
  left_join(edges_in_comp, by = "orig_comp_id") |>
  mutate(n_edges = coalesce(n_edges, 0L))

if (nrow(comp_sizes) == 0L) fail("No components found; cannot split.")

n_comps <- nrow(comp_sizes)
logi("Detected %d connected components (median nodes: %d).",
     n_comps, stats::median(comp_sizes$n_nodes))

## ---------------- Community detection in large components ----------------
# Assign each node to a "block_id": either a Louvain community (inside a large component)
# or the whole small component acting as one block.
block_id_of_node <- integer(length(nodes))
names(block_id_of_node) <- nodes

block_meta <- list()   # will bind into a data frame
block_counter <- 0L

for (i in seq_len(n_comps)) {
  ocid <- comp_sizes$orig_comp_id[i]
  sub_nodes <- nodes[comp_id == ocid]
  sg <- induced_subgraph(g, vids = sub_nodes)
  nN <- length(sub_nodes)
  nE <- gsize(sg)
  
  split_here <- (nN >= giant_min_nodes) || (nE >= giant_min_edges)
  
  if (split_here) {
    # Louvain community detection (weights optional; set if you have one)
    comm <- cluster_louvain(sg)
    memb <- membership(comm)  # named by sub_nodes
    groups <- split(names(memb), memb)
    
    for (grp in groups) {
      block_counter <- block_counter + 1L
      block_id_of_node[grp] <- block_counter
      block_meta[[length(block_meta) + 1L]] <- list(
        block_id      = block_counter,
        orig_comp_id  = ocid,
        block_type    = "community",
        n_nodes       = length(grp)
      )
    }
  } else {
    block_counter <- block_counter + 1L
    block_id_of_node[sub_nodes] <- block_counter
    block_meta[[length(block_meta) + 1L]] <- list(
      block_id      = block_counter,
      orig_comp_id  = ocid,
      block_type    = "component",
      n_nodes       = nN
    )
  }
}

block_meta <- bind_rows(block_meta) |> arrange(block_id)
n_blocks <- nrow(block_meta)
logi("Formed %d blocks (micro-communities + small components).", n_blocks)

## ---------------- Block-level stats ----------------
# Tag edges with block endpoints
df_blocks <- df %>%
  mutate(
    block1 = block_id_of_node[match(kw1, nodes)],
    block2 = block_id_of_node[match(kw2, nodes)]
  )

# Intra-block edges (eligible to be kept regardless of fold packing)
intra_edges <- df_blocks %>% filter(block1 == block2) %>%
  mutate(block_id = block1) %>%
  select(-block1, -block2)

# Cross-block edges (can be kept ONLY if both blocks end up in the same fold; we choose
# to NOT rely on that and optimize packing using intra stats for stability)
cross_edges <- df_blocks %>% filter(block1 != block2)

# Block edge counts
blk_edge_counts <- intra_edges %>% count(block_id, name = "n_edges_intra")
blk_cross_counts <- bind_rows(
  cross_edges %>% count(block1, name = "n_edges_cross_out") %>% rename(block_id = block1),
  cross_edges %>% count(block2, name = "n_edges_cross_in")  %>% rename(block_id = block2)
) %>%
  group_by(block_id) %>%
  summarise(n_edges_cut_touching = sum(n_edges_cross_out %||% 0L, n_edges_cross_in %||% 0L), .groups = "drop")

block_stats <- block_meta %>%
  left_join(blk_edge_counts, by = "block_id") %>%
  left_join(blk_cross_counts, by = "block_id") %>%
  mutate(
    n_edges_intra = coalesce(n_edges_intra, 0L),
    n_edges_cut_touching = coalesce(n_edges_cut_touching, 0L),
    weight = if (balance_by == "edges") n_edges_intra else n_nodes
  )

if (all(block_stats$weight == 0L)) {
  logi("All block %s are zero; falling back to node counts.", balance_by)
  block_stats$weight <- block_stats$n_nodes
}

## ---------------- Optional: score histogram per block ----------------
score_breaks <- NULL
B <- 0L
if (!is.null(score_col)) {
  all_scores <- suppressWarnings(df[[score_col]])
  all_scores <- all_scores[is.finite(all_scores)]
  if (length(all_scores) >= 2L) {
    # Use quantile cuts for balanced targets
    probs <- seq(0, 1, length.out = n_score_bins + 1)
    q <- unique(stats::quantile(all_scores, probs = probs, na.rm = TRUE, type = 7))
    # Ensure at least 2 distinct edges for breaks
    if (length(q) >= 2L) {
      score_breaks <- q
      B <- length(score_breaks) - 1L
    }
  }
}

# Per-block bin counts from intra-block edges only
if (!is.null(score_breaks)) {
  intra_bins <- intra_edges %>%
    mutate(bin = cut(.data[[score_col]], breaks = score_breaks, include.lowest = TRUE, right = TRUE)) %>%
    count(block_id, bin, name = "n_bin") %>%
    mutate(bin = as.integer(bin)) %>%
    tidyr::complete(block_id, bin = seq_len(B), fill = list(n_bin = 0L)) %>%
    arrange(block_id, bin)
  
  block_bins <- intra_bins %>%
    pivot_wider(names_from = bin, values_from = n_bin, names_prefix = "bin_", values_fill = 0L)
  
  block_stats <- block_stats %>%
    left_join(block_bins, by = "block_id")
  
  # Fill missing bin columns with zeros
  for (b in seq_len(B)) {
    colb <- paste0("bin_", b)
    if (!(colb %in% names(block_stats))) block_stats[[colb]] <- 0L
  }
}

## ---------------- Greedy K-way packing of blocks (incremental kept-edges) ----------------
# Cross-block edge weights between blocks (undirected)
cross_pairs <- cross_edges %>%
  transmute(b1 = pmin(block1, block2), b2 = pmax(block1, block2)) %>%
  count(b1, b2, name = "w")

# Build neighbor lists: for each block, which other blocks and how many edges to them
nb_tbl <- bind_rows(
  cross_pairs %>% transmute(block_id = b1, nb = b2, w),
  cross_pairs %>% transmute(block_id = b2, nb = b1, w)
)
neigh <- split(nb_tbl[, c("nb","w")], nb_tbl$block_id)
# Ensure every block has an entry
empty_blocks <- setdiff(block_stats$block_id, as.integer(names(neigh)))
if (length(empty_blocks)) {
  for (bb in empty_blocks) neigh[[as.character(bb)]] <- tibble::tibble(nb = integer(), w = integer())
}

# Cross-degree (sum of cross edges touching the block) and strength for seeding
cross_deg <- vapply(block_stats$block_id, function(b) {
  sum(neigh[[as.character(b)]]$w)
}, numeric(1))
block_stats$cross_degree <- cross_deg
block_stats$strength <- block_stats$n_edges_intra + block_stats$cross_degree

# Helper: map block_id -> row index in block_stats
row_of_block <- setNames(seq_len(nrow(block_stats)), as.character(block_stats$block_id))

fold_ids <- seq_len(k_folds)
k <- k_folds
fold_weight <- rep(0, k)                          # predicted kept edges per fold (intra + captured cross)
fold_bins   <- if (B > 0) matrix(0L, nrow = k, ncol = B) else NULL
assigned_fold <- integer(nrow(block_stats))       # 0 = unassigned; else fold id
fold_blocks   <- vector("list", k)                # store block_ids per fold

# Seeding: put the top-k strongest blocks into distinct folds
seed_order <- order(-block_stats$strength, -block_stats$n_nodes)
seed_take  <- head(seed_order, k)
for (i in seq_along(seed_take)) {
  idx <- seed_take[i]
  f <- i
  b_id <- block_stats$block_id[idx]
  inc <- block_stats$n_edges_intra[idx]           # no neighbors yet in fold
  fold_weight[f] <- fold_weight[f] + inc
  assigned_fold[idx] <- f
  fold_blocks[[f]] <- c(fold_blocks[[f]], b_id)
  if (B > 0) {
    bvec <- as.integer(block_stats[idx, paste0("bin_", seq_len(B))])
    fold_bins[f, ] <- fold_bins[f, ] + bvec
  }
}

# Function to compute incremental kept edges if placing block idx into fold f
inc_gain <- function(idx, f) {
  b_id <- block_stats$block_id[idx]
  nb_df <- neigh[[as.character(b_id)]]
  inc <- block_stats$n_edges_intra[idx]
  if (nrow(nb_df)) {
    nb_rows <- row_of_block[as.character(nb_df$nb)]
    # Which neighbors are already in this fold?
    hits <- which(assigned_fold[nb_rows] == f)
    if (length(hits)) inc <- inc + sum(nb_df$w[hits])
  }
  inc
}

# Assign remaining blocks in descending strength
remaining <- setdiff(seq_len(nrow(block_stats)), seed_take)
remaining <- remaining[order(-block_stats$strength[remaining], -block_stats$n_nodes[remaining])]

for (idx in remaining) {
  # For each fold, compute resulting weight if we place this block there
  incs <- vapply(fold_ids, function(f) inc_gain(idx, f), numeric(1))
  res_weights <- fold_weight + incs
  min_res <- min(res_weights)
  
  # Candidates that minimize resulting weight (balance first)
  cand <- which(res_weights <= min_res + 1e-9)
  
  # Among candidates, prefer better score-bin balance (if enabled)
  if (B > 0) {
    bvec <- as.integer(block_stats[idx, paste0("bin_", seq_len(B))])
    bin_cost <- vapply(cand, function(f) {
      nb <- fold_bins[f, ] + bvec
      sum(abs(nb - target_bins_per_fold) / pmax(target_bins_per_fold, 1))
    }, numeric(1))
    cand <- cand[bin_cost == min(bin_cost)]
  }
  
  # Final tie-break: capture the most cross edges (maximize inc)
  if (length(cand) > 1L) {
    best_inc <- which.max(incs[cand])
    f <- cand[best_inc]
  } else {
    f <- cand[1]
  }
  
  # Commit
  b_id <- block_stats$block_id[idx]
  fold_weight[f] <- fold_weight[f] + incs[f]
  assigned_fold[idx] <- f
  fold_blocks[[f]] <- c(fold_blocks[[f]], b_id)
  if (B > 0) {
    bvec <- as.integer(block_stats[idx, paste0("bin_", seq_len(B))])
    fold_bins[f, ] <- fold_bins[f, ] + bvec
  }
}

block_stats$fold <- assigned_fold


## ---------------- Keyword & edge assignment ----------------
kw_assign <- tibble::tibble(
  keyword   = nodes,
  orig_comp_id = comp_id,
  block_id  = block_id_of_node[nodes],
  fold      = block_stats$fold[match(block_id, block_stats$block_id)]
) %>% arrange(keyword)

# Sanity check
dup_kw <- kw_assign %>% count(keyword) %>% filter(n > 1)
if (nrow(dup_kw) > 0L) fail("Duplicate keyword assignment detected.")

# Join folds onto edges; keep ONLY edges whose endpoints are in the same fold
df_with_folds <- df %>%
  left_join(select(kw_assign, keyword, fold), by = c("kw1" = "keyword")) %>%
  rename(fold1 = fold) %>%
  left_join(select(kw_assign, keyword, fold), by = c("kw2" = "keyword")) %>%
  rename(fold2 = fold)

n_cross_fold <- nrow(df_with_folds %>% filter(fold1 != fold2))
cut_ratio <- n_cross_fold / nrow(df_with_folds)

pairs_with_fold <- df_with_folds %>%
  filter(fold1 == fold2) %>%
  transmute(kw1, kw2, !!!syms(extra_cols), fold = fold1)

## ---------------- Outputs ----------------
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Block table (acts as "components" file, with provenance)
write_csv(
  block_stats %>%
    transmute(
      comp_id   = block_id,          # keep column name stable for downstream code
      orig_comp_id,
      block_type,
      n_nodes,
      n_edges = n_edges_intra,
      n_edges_cut_touching,
      weight,
      fold
    ) %>% arrange(desc(weight), comp_id),
  out_comp_csv
)

write_csv(kw_assign %>% select(keyword, fold), out_kw_csv)
write_csv(pairs_with_fold, out_pairs_csv)

summary_tbl <- pairs_with_fold %>%
  count(fold, name = "n_edges") %>%
  tidyr::complete(fold = fold_ids, fill = list(n_edges = 0L)) %>%
  mutate(weight = as.integer(fold_weight[as.integer(fold)])) %>%
  mutate(
    edges_share  = ifelse(sum(n_edges) > 0, n_edges / sum(n_edges), NA_real_),
    weight_share = ifelse(sum(weight) > 0, weight / sum(weight), NA_real_)
  ) %>%
  arrange(fold)

write_csv(summary_tbl, out_summary_csv)

## ---------------- Logging ----------------
logi("K-fold split complete: K=%d, balance_by=%s, lambda_bins=%.3f", k_folds, balance_by, lambda_bins)
logi("Blocks formed: %d (communities + small components).", n_blocks)
logi("Fold weights: %s", paste(summary_tbl$weight, collapse = ", "))
logi("Fold kept-edge counts: %s", paste(summary_tbl$n_edges, collapse = ", "))
logi("Cross-fold edges dropped: %d of %d (cut ratio = %.2f%%).",
     n_cross_fold, nrow(df_with_folds), 100 * cut_ratio)
logi("Wrote:\n  - %s\n  - %s\n  - %s\n  - %s",
     out_pairs_csv, out_comp_csv, out_kw_csv, out_summary_csv)
logi("Done.")
