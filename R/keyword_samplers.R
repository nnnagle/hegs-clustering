# keyword_samplers.R
# -------------------------------------------------------
# All functions return a 2-col integer matrix of indices (i, j) with i<j.
# Indices refer to the CURRENT active keyword subset (rows of kw_sub / sim / C).
#
# Common args:
#   n               : number of pairs requested
#   kids            : character vector of keyword IDs aligned to row order
#   used_pair_keys  : character vector of unordered "kid1||kid2" keys to exclude
#                     If NULL, skip exclusion (faster).
#
# Additional:
#   sim             : dense cosine similarity matrix, same order as kids
#   C               : sparse co-occurrence (dgCMatrix), same order as kids
#   target, tol     : distance-midpoint params
#   min_shared, max_sim : cooccur filters
#   min_sim, max_shared : hard-negative filters
#
# Notes:
# - These implementations AVOID materializing all pairs (no combn()).
# - Co-occur samplers sample directly from sparse structure (no full Ts expansion).
# - All samplers stop early if not enough valid pairs exist.

# ----------------- helpers -----------------

pair_key <- function(a, b) paste(ifelse(a < b, a, b), ifelse(a < b, b, a), sep = "||")

# TRUE if a pair should be kept (not already used)
keep_pair <- function(i, j, kids, used_pair_keys) {
  if (is.null(used_pair_keys) || !length(used_pair_keys)) return(TRUE)
  !(pair_key(kids[i], kids[j]) %in% used_pair_keys)
}

# Append a pair if valid, ensuring i<j and not used; returns updated list
append_if_ok <- function(out_i, out_j, i, j, kids, used_pair_keys) {
  if (i == j) return(list(out_i = out_i, out_j = out_j, added = FALSE))
  if (j < i) { tmp <- i; i <- j; j <- tmp }
  if (!keep_pair(i, j, kids, used_pair_keys)) {
    return(list(out_i = out_i, out_j = out_j, added = FALSE))
  }
  list(out_i = c(out_i, i), out_j = c(out_j, j), added = TRUE)
}

# ----------------- 1) Random sampler -----------------
# Draws pairs by random indices; rejects used/duplicates; no combn().
sample_pair_random_n <- function(n_kw, n, kids, used_pair_keys = NULL,
                                 max_draws_multiplier = 50L) {
  if (n_kw < 2L || n < 1L) return(matrix(NA_integer_, 0, 2))
  want <- as.integer(n)
  out_i <- integer(0); out_j <- integer(0)
  seen <- new.env(hash = TRUE, parent = emptyenv())
  
  max_draws <- max(want * max_draws_multiplier, want * 5L)
  draws <- 0L
  
  while (length(out_i) < want && draws < max_draws) {
    draws <- draws + 1L
    i <- sample.int(n_kw, 1L)
    j <- sample.int(n_kw, 1L)
    if (i == j) next
    if (j < i) { tmp <- i; i <- j; j <- tmp }
    
    key <- paste(i, j, sep = "-")
    if (!is.null(seen[[key]])) next
    if (!keep_pair(i, j, kids, used_pair_keys)) next
    
    seen[[key]] <- TRUE
    out_i <- c(out_i, i); out_j <- c(out_j, j)
  }
  
  if (!length(out_i)) return(matrix(NA_integer_, 0, 2))
  cbind(out_i, out_j)
}

# -------- 2) Distance-midpoint sampler (no combn) --------
sample_pair_distance_mid_n <- function(sim, n, kids, target, tol, used_pair_keys = NULL) {
  n_kw <- length(kids)
  pairs <- matrix(NA_integer_, nrow = 0, ncol = 2)
  sim_target_min <- target - tol
  sim_target_max <- target + tol
  
  # Shuffle keyword indices so we don't bias toward early IDs
  kw_indices <- sample(seq_len(n_kw))
  
  for (i in kw_indices) {
    if (nrow(pairs) >= n) break
    
    # Candidates for the second keyword: within similarity band
    cand_idx <- which(sim[i, ] >= sim_target_min &
                        sim[i, ] <= sim_target_max &
                        seq_len(n_kw) != i)
    
    if (length(cand_idx) == 0) next
    
    # Pick one candidate at random
    j <- sample(cand_idx, 1)
    
    # Form key for used-pair check
    pk <- paste(sort(c(kids[i], kids[j])), collapse = "||")
    if (!is.null(used_pair_keys) && pk %in% used_pair_keys) next
    
    pairs <- rbind(pairs, c(i, j))
  }
  
  # Return only up to n rows
  if (nrow(pairs) > n) pairs <- pairs[seq_len(n), , drop = FALSE]
  pairs
}

# ---------- Sparse helpers: sample from a dgCMatrix ----------
# Randomly sample column j, then one eligible i from its nonzeros.
.sample_from_sparse <- function(C, n, value_filter_fn, kids, used_pair_keys,
                                sim = NULL, sim_filter_fn = NULL,
                                ensure_upper_tri = TRUE,
                                max_trials = 50000L) {
  stopifnot(inherits(C, "dgCMatrix"))
  p <- C@p        # column pointers (length ncol+1)
  i_idx <- C@i    # row indices (0-based)
  x_val <- C@x    # nonzero values
  n_kw <- ncol(C)
  
  out_i <- integer(0); out_j <- integer(0)
  seen <- new.env(hash = TRUE, parent = emptyenv())
  
  trials <- 0L
  while (length(out_i) < n && trials < max_trials) {
    trials <- trials + 1L
    j <- sample.int(n_kw, 1L) - 1L           # 0-based for p
    start <- p[j + 1L] + 1L                  # 1-based into i_idx/x_val
    end   <- p[j + 2L]
    if (end < start) next                    # empty column
    # Pick a random nonzero in this column
    k <- sample.int(end - start + 1L, 1L) + start - 1L
    i0 <- i_idx[k]                           # 0-based row
    i  <- i0 + 1L                            # 1-based row
    jj <- j + 1L                             # 1-based col
    
    # Enforce upper triangle if requested
    if (ensure_upper_tri && i >= jj) next
    
    val <- x_val[k]
    if (!value_filter_fn(val)) next
    
    # Optional sim filter
    if (!is.null(sim) && !is.null(sim_filter_fn)) {
      if (!sim_filter_fn(sim[i, jj])) next
    }
    
    # Dedup & used check
    key <- paste(i, jj, sep = "-")
    if (!is.null(seen[[key]])) next
    if (!keep_pair(i, jj, kids, used_pair_keys)) next
    
    seen[[key]] <- TRUE
    out_i <- c(out_i, i); out_j <- c(out_j, jj)
  }
  
  if (!length(out_i)) return(matrix(NA_integer_, 0, 2))
  cbind(out_i, out_j)
}

# ------------- 3) Co-occurrence sampler (sparse) -------------
sample_pair_cooccur_sparse_n <- function(n, C, sim, kids,
                                         min_shared = 1L, max_sim = 0.9,
                                         used_pair_keys = NULL,
                                         max_trials = 50000L) {
  if (nrow(C) < 2L || n < 1L) return(matrix(NA_integer_, 0, 2))
  value_filter <- function(v) v >= min_shared
  sim_filter   <- function(s) s <= max_sim
  .sample_from_sparse(C, n, value_filter, kids, used_pair_keys,
                      sim = sim, sim_filter_fn = sim_filter,
                      ensure_upper_tri = TRUE, max_trials = max_trials)
}

# -------- 4) Hard-negative sampler (sparse) --------
# -------- 4) Hard-negative sampler (supports zeros) --------
# Samples pairs with high similarity AND low/zero co-occurrence.
# -------- 4) Hard-negative sampler (sparse) — co-occur-like interface --------
# Returns pairs (i<j) with:
#   sim[i, j] >= min_sim  AND  C[i, j] <= max_shared   (zeros allowed)
# Signature mirrors sample_pair_cooccur_sparse_n(...)
sample_pair_hard_negative_sparse_n <- function(
    n, C, sim, kids,
    min_sim = 0.7, max_shared = 0L,
    used_pair_keys = NULL,
    max_trials = 50000L   # kept for interface parity; not used directly
) {
  stopifnot(nrow(C) == ncol(C), nrow(sim) == ncol(sim), nrow(sim) == nrow(C))
  N <- nrow(sim)
  if (N < 2L || n < 1L) return(matrix(NA_integer_, 0, 2))
  
  want <- as.integer(n)
  out_i <- integer(0L); out_j <- integer(0L)
  
  # 1) Similarity mask (upper triangle)
  sim_mask <- sim >= min_sim
  sim_mask[lower.tri(sim_mask, diag = TRUE)] <- FALSE
  
  # 2) Optional used set (hash env for O(1) membership checks)
  used_env <- if (is.null(used_pair_keys) || !length(used_pair_keys)) NULL else {
    e <- new.env(parent = emptyenv(), hash = TRUE)
    for (k in used_pair_keys) e[[k]] <- TRUE
    e
  }
  
  # Sweep anchors in random order; vectorized filtering per anchor
  anchors <- sample.int(N, N, replace = FALSE)
  for (i in anchors) {
    if (length(out_i) >= want) break
    
    # Candidates by sim & i<j
    cand_js <- which(sim_mask[i, ])
    if (!length(cand_js)) next
    
    # Co-occur filter: C[i, j] <= max_shared  (zeros included)
    cvals <- C[i, cand_js, drop = TRUE]
    keep_c <- cvals <= max_shared
    if (!any(keep_c)) next
    cand_js <- cand_js[keep_c]
    
    # Exclude used pairs if provided
    if (!is.null(used_env)) {
      keys <- paste(
        pmin(kids[i], kids[cand_js]),
        pmax(kids[i], kids[cand_js]),
        sep = "||"
      )
      not_used <- !vapply(keys, function(k) !is.null(used_env[[k]]), logical(1))
      if (!any(not_used)) next
      cand_js <- cand_js[not_used]
    }
    
    if (!length(cand_js)) next
    
    # Pick one partner for this i to avoid reusing the same first keyword a lot
    j <- if (length(cand_js) > 1L) sample(cand_js, 1L) else cand_js
    
    out_i <- c(out_i, i)
    out_j <- c(out_j, j)
    
    if (length(out_i) >= want) break
  }
  
  if (!length(out_i)) return(matrix(NA_integer_, 0, 2))
  if (length(out_i) > want) {
    keep <- seq_len(want)
    out_i <- out_i[keep]; out_j <- out_j[keep]
  }
  cbind(out_i, out_j)
}
