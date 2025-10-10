# Make utils available to tests when this is not a package project.
# testthat runs with working dir = tests/testthat/, so go up one level.
utils_path <- normalizePath(file.path("../..", "shiny", "utils.R"), mustWork = TRUE)
source(utils_path, local = TRUE)
