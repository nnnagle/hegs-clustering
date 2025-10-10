# utils.R — Small, pure helpers used across modules
# -------------------------------------------------
# Purpose:
#   Centralizes stateless utilities (string cleanup, normalization, keying, and
#   HTML highlighting) so modules stay lean and testable.
#
# Exported helpers:
#   - pair_key(a, b)
#       Build an unordered lexical key "A||B" from two character vectors.
#       Use to dedupe/lookup pairs regardless of order.
#
#   - nzchr(x)
#       Vector-safe coalescer to character: NULL/NA/length-0 → "" ; otherwise as.character(x).
#
#   - norm_init(x)
#       Normalize rater initials: trim whitespace, uppercase, coalesce empties via nzchr().
#
#   - renorm(M)
#       Row-wise L2 normalization of a numeric matrix; zero rows remain zero (safe).
#
#   - highlight_phrase(text, phrase)
#       Case-insensitive literal match of `phrase` in `text`; wraps matches with <mark>.
#       Uses stringr::str_escape() to treat `phrase` literally.
#
#   - compute_used_keys(phr_to_kid, rows_df)
#       Given a named character vector `phr_to_kid` (names = phrases, values = kid)
#       and a data frame with columns `kw1`, `kw2`, returns a unique character
#       vector of unordered pair keys via pair_key(). Returns NULL if none.
#
# Inputs / Contracts:
#   - `phrase` values are plain strings (not regex); escaping handled internally.
#   - `phr_to_kid` must contain all phrases referenced by rows_df$kw1/kw2 when
#     you expect keys back; missing mappings are dropped.
#
# Dependencies:
#   - Base R only, except `highlight_phrase()` which requires {stringr}.
#
# Notes:
#   - All helpers are vectorized where sensible.
#   - Keep this file free of Shiny/reactivity to preserve testability.
#
# Example (compute_used_keys):
#   map <- c("alpha" = "K1", "beta" = "K2", "gamma" = "K3")
#   df  <- data.frame(kw1 = c("alpha","beta"), kw2 = c("gamma","alpha"))
#   compute_used_keys(map, df)  # → c("K1||K3", "K1||K2")
#
# Change log:
#   - v1: initial extraction from monolithic app.R
# -----------------------------------------------------------------------------------
# shiny/utils.R

# Unordered key from two strings (A||B with lexical order)
pair_key <- function(a, b) paste(ifelse(a < b, a, b), ifelse(a < b, b, a), sep = "||")

# Vector-safe empty-coalescer
# Vector-safe coalescer to character:
# - NULL or length-0  -> ""  (scalar length 1)
# - NA values         -> ""  (elementwise)
nzchr <- function(x) {
  if (is.null(x)) return("")
  if (length(x) == 0L) return("")
  na <- is.na(x)
  x <- as.character(x)
  x[na] <- ""
  x
}


# Normalizer for rater initials (vector-safe)
norm_init <- function(x) toupper(trimws(nzchr(x)))

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
  out <- character(0); last <- 1L
  for (i in seq_len(nrow(locs))) {
    s <- locs[i,1]; e <- locs[i,2]
    out <- c(out, substr(text, last, s - 1L), "<mark>", substr(text, s, e), "</mark>")
    last <- e + 1L
  }
  out <- c(out, substr(text, last, nchar(text)))
  paste0(out, collapse = "")
}

# Compute used pair keys from a (kw1, kw2) df, given a phrase->kid map
compute_used_keys <- function(phr_to_kid, rows_df) {
  if (is.null(rows_df) || nrow(rows_df) == 0) return(NULL)
  a <- unname(phr_to_kid[rows_df$kw1])
  b <- unname(phr_to_kid[rows_df$kw2])
  keep <- !is.na(a) & !is.na(b)
  if (!any(keep)) return(NULL)
  unique(pair_key(a[keep], b[keep]))
}
