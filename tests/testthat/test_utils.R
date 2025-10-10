# Tests for shiny/utils.R
# Uses testthat 3e. Run with: testthat::test_dir("tests/testthat")
library(testthat)
library(stringr)

test_that("pair_key() is order-invariant and vectorized", {
  expect_equal(pair_key("A", "B"), "A||B")
  expect_equal(pair_key("B", "A"), "A||B")   # order shouldn't matter
  expect_equal(pair_key("A", "A"), "A||A")
  a <- c("z", "a", "b")
  b <- c("a", "z", "b")
  expect_equal(
    pair_key(a, b),
    c("a||z", "a||z", "b||b")
  )
})

test_that("nzchr() is vector-safe and coalesces empties", {
  expect_identical(nzchr(NULL), "")
  expect_identical(nzchr(character(0)), "")
  expect_identical(nzchr(NA_character_), "")
  expect_identical(nzchr(NA), "")  # numeric NA also coalesces
  
  v <- c("a", NA, "", "  ")
  # nzchr does not trim; it just coalesces and casts
  expect_identical(nzchr(v), c("a", "", "", "  "))
})

test_that("norm_init() trims + uppercases and is vectorized", {
  v <- c(" ab ", "Cd", NA, "", "  ef  ")
  expect_identical(norm_init(v), c("AB", "CD", "", "", "EF"))
})

test_that("renorm() returns unit-length rows and keeps zero rows zero", {
  M <- rbind(
    c(3, 4, 0),   # norm 5
    c(0, 0, 0),   # zero row
    c(1, 1, 1)    # norm sqrt(3)
  )
  R <- renorm(M)
  
  # Row norms ~ 1, 0, 1
  rn <- sqrt(rowSums(R^2))
  expect_true(abs(rn[1] - 1) < 1e-12)
  expect_true(abs(rn[2] - 0) < 1e-12)
  expect_true(abs(rn[3] - 1) < 1e-12)
  
  # First row specifically 3/5, 4/5, 0
  expect_equal(R[1, ], c(0.6, 0.8, 0.0), tolerance = 1e-12)
})

test_that("highlight_phrase() wraps literal, case-insensitive matches with <mark>", {
  txt <- "Cats, dogs, and DOGS."
  out <- highlight_phrase(txt, "dogs")
  expect_true(grepl("<mark>dogs</mark>", out, fixed = TRUE))
  expect_true(grepl("<mark>DOGS</mark>", out, fixed = TRUE))
  
  # Literal special chars should be escaped (e.g., C++, plus signs, dots)
  txt2 <- "We teach C++ and not C+."
  out2 <- highlight_phrase(txt2, "C++")
  expect_true(grepl("<mark>C\\+\\+</mark>", out2))  # regex view of the result
  # If you prefer a fixed-string check, strip the tags and compare:
  expect_true(grepl("<mark>C++</mark>", out2, fixed = TRUE))
  
  # No changes when phrase is empty or NA; NA text returns NA
  expect_identical(highlight_phrase("hello", ""), "hello")
  expect_identical(highlight_phrase("hello", NA_character_), "hello")
  expect_true(is.na(highlight_phrase(NA_character_, "hi")))
})

test_that("compute_used_keys() returns unique unordered keys and drops unknown phrases", {
  phr_to_kid <- c("alpha" = "K1", "beta" = "K2", "gamma" = "K3")
  rows <- data.frame(
    kw1 = c("alpha", "beta",  "missing", "alpha"),
    kw2 = c("gamma", "alpha", "beta",    "gamma"),
    stringsAsFactors = FALSE
  )
  
  keys <- compute_used_keys(phr_to_kid, rows)
  # Expected: alpha-gamma, alpha-beta (unordered), unique
  expect_setequal(keys, c("K1||K3", "K1||K2"))
  
  # If nothing matches, return NULL (not character(0))
  rows2 <- data.frame(kw1 = "missing", kw2 = "also-missing")
  expect_null(compute_used_keys(phr_to_kid, rows2))
  
  # Empty df → NULL
  expect_null(compute_used_keys(phr_to_kid, rows[0, ]))
})
