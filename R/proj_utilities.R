## proj_utilities.R

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")

logi <- function(...) cat(sprintf("[%s] ", timestamp()), sprintf(...), "\n")

ensure_dir <- function(path) dir.create(path, recursive = TRUE, showWarnings = FALSE)

require_cols <- function(df, cols, where) {
  miss <- setdiff(cols, names(df))
  if (length(miss)) stop(sprintf("%s missing columns: %s", where, paste(miss, collapse = ", ")))
}

as_chr <- function(x) as.character(replace(x, is.na(x), ""))
