# ------------------------------------------------------------------------------
# Script: 10_HEGS_keyword_candidates.R
#
# Purpose:
#   Extract candidate keyword phrases from cleaned NSF HEGS abstracts using
#   (1) noun phrases via spaCy, and (2) collocations via text2vec.
#   For each phrase, record which award_ids it appears in and basic metadata.
#
# Inputs:
#   data/interim/HEGS_clean_df.csv
#     - Must contain columns: 'award_id' and 'abstract_clean'
#
# Outputs:
#   data/interim/keyword_candidates.csv
#   data/interim/shiny_keyword_candidates.csv   (same content, separate name)
#
# Algorithm:
#   1) Load cleaned abstracts (CSV), validate required columns.
#   2) Extract noun phrases with spaCy (spacyr::spacy_extract_nounphrases),
#      map phrase → set(award_id), keep phrases with doc_count >= min_doc_count.
#   3) Build collocations with text2vec Collocations; map term → set(award_id),
#      trim leading/trailing stopwords; keep terms with doc_count >= min_doc_count.
#   4) Union noun phrases + collocations; normalize text; remove phrases that
#      begin or end with stopwords; keep multi-word phrases only.
#   5) For each phrase, store unique award_ids (list), doc_count, and blank
#      metadata fields (omit/method/thematic), plus a string ID 'kid'.
#   6) Write results as CSVs.
#
# Notes:
#   - spaCy in R requires python model (e.g., en_core_web_sm). If missing,
#     install in your python env and set 'spacy_model' below.
#   - Filtering is controlled by 'min_doc_count' (default 5).
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(readr)      # read_csv, write_csv
  library(dplyr)      # mutate, group_by, summarise, filter, arrange
  library(stringr)    # str_*
  library(spacyr)     # spaCy noun phrases
  library(text2vec)   # collocations
  library(stopwords)  # stopword lists
})

## =============================
## PARAMETERS (edit here)
## =============================

# Paths
input_file   <- "data/interim/HEGS_clean_df.csv"
out_dir      <- "data/interim"
out_file    <- file.path(out_dir, "keyword_candidates.csv")

# Phrase frequency threshold (minimum # of distinct award_ids)
min_doc_count <- 5L

# spaCy model (set to NULL to use default configured model)
spacy_model <- NULL     # e.g., "en_core_web_sm"

# Collocation thresholds
collocation_count_min <- 5L
collocation_pmi_min   <- 2.0

# Stopword handling
stop_lang <- "en"       # language for stopwords()

## =============================
## UTILITIES / LOGGING
## =============================

timestamp <- function() format(Sys.time(), "%Y-%m-%d %H:%M:%S")
logi <- function(...) { cat(sprintf("[%s] ", timestamp()), sprintf(...), "\n") }
ensure_dir <- function(path) dir.create(path, recursive = TRUE, showWarnings = FALSE)

# Trim a multi-word term’s leading/trailing stopwords
trim_term <- function(term, stopwords_vec) {
  words <- str_split(term, " ", simplify = FALSE)[[1]]
  while (length(words) > 0 && tolower(words[1]) %in% stopwords_vec) {
    words <- words[-1]
  }
  while (length(words) > 0 && tolower(words[length(words)]) %in% stopwords_vec) {
    words <- words[-length(words)]
  }
  str_squish(paste(words, collapse = " "))
}

## =============================
## LOAD INPUT & VALIDATE
## =============================

if (!file.exists(input_file)) stop(sprintf("Input not found: %s", input_file))
ensure_dir(out_dir)

logi("Reading cleaned abstracts: %s", input_file)
df <- read_csv(input_file, show_col_types = FALSE)

req_cols <- c("award_id", "abstract_clean")
missing_cols <- setdiff(req_cols, names(df))
if (length(missing_cols)) {
  stop(sprintf("Input is missing required columns: %s", paste(missing_cols, collapse = ", ")))
}

df <- df %>%
  filter(!is.na(abstract_clean), nchar(abstract_clean) > 0)

n_docs <- nrow(df)
logi("Loaded %d abstracts after filtering.", n_docs)

if (n_docs == 0L) stop("No abstracts to process after filtering.")

## =============================
## NOUN PHRASES (spaCy)
## =============================

logi("Initializing spaCy...")
spacy_inited <- FALSE
try({
  if (is.null(spacy_model)) {
    spacy_initialize()
  } else {
    spacy_initialize(model = spacy_model)
  }
  spacy_inited <- TRUE
}, silent = TRUE)

if (!spacy_inited) {
  stop("spaCy failed to initialize. Ensure a model (e.g., en_core_web_sm) is installed in your python env.")
}

logi("Extracting noun phrases with spaCy...")
# spacyr uses doc_id like "text1", "text2", ...
# Reconstruct mapping to award_id by position
spacy_phrases <- spacy_extract_nounphrases(df$abstract_clean) %>%
  mutate(doc_index = as.integer(str_extract(doc_id, "\\d+")),
         award_id  = df$award_id[doc_index])

# Clean and tally noun phrases
noun_phrase_counts <- spacy_phrases %>%
  transmute(nounphrase = text,
            award_id   = award_id) %>%
  mutate(
    nounphrase = str_trim(nounphrase),
    nounphrase = tolower(nounphrase),
    nounphrase = str_remove(nounphrase, "^(a|an|the|this|that|these|those|which|where)\\s+"),
    nounphrase = str_replace_all(nounphrase, "[^a-z0-9\\- ]", ""),  # keep letters, digits, hyphen, space
    nounphrase = str_squish(nounphrase),
    word_count = str_count(nounphrase, "\\S+")
  ) %>%
  filter(word_count >= 2) %>%
  group_by(nounphrase) %>%
  summarise(
    noun_count = n_distinct(award_id),
    award_ids  = list(sort(unique(award_id))),
    .groups = "drop"
  ) %>%
  filter(lengths(award_ids) >= min_doc_count)

logi("Noun-phrase candidates: %d (doc_count >= %d).",
     nrow(noun_phrase_counts), min_doc_count)

spacy_finalize()

## =============================
## COLLOCATIONS (text2vec)
## =============================

logi("Training collocations with text2vec...")
preprocessor <- function(x) {
  # Lowercase + remove punctuation (keep spaces)
  gsub("[^[:alnum:]\\s]", " ", tolower(x))
}

tokens <- word_tokenizer(preprocessor(df$abstract_clean))
it <- itoken(tokens, progressbar = FALSE)

colloc_model <- Collocations$new(
  collocation_count_min = as.integer(collocation_count_min),
  pmi_min               = collocation_pmi_min
)
colloc_model$fit(it, n_iter = 2)

# Build doc_texts for regex matching of terms back to documents
doc_texts <- vapply(tokens, paste, collapse = " ", FUN.VALUE = character(1))

# Extract & clean collocation stats
collocs_clean <- colloc_model$collocation_stat %>%
  mutate(term = paste(prefix, suffix, sep = " ") %>% str_replace_all("_", " ")) %>%
  arrange(desc(llr))

# Trim leading/trailing stopwords from collocations
sw <- tolower(gsub("'", "", stopwords(language = stop_lang)))
collocs_clean$term <- vapply(collocs_clean$term, trim_term, stopwords_vec = sw, FUN.VALUE = character(1))
collocs_clean$term <- str_squish(collocs_clean$term)

# Map terms to award_ids via regex search over doc_texts
logi("Mapping collocations back to documents...")
term_doc_matches <- lapply(collocs_clean$term, function(term) {
  if (is.na(term) || term == "") return(integer(0))
  # Regex: allow whitespace between words, honor word boundaries
  pattern <- paste0("\\b", gsub(" ", "\\\\s+", term), "\\b")
  which(grepl(pattern, doc_texts, perl = TRUE))
})

# Convert doc indices to award_ids
term_award_ids <- lapply(term_doc_matches, function(ix) {
  if (length(ix) == 0) character(0) else df$award_id[ix]
})

collocs_clean$award_ids <- term_award_ids
collocs_clean$n <- lengths(collocs_clean$award_ids)

# Keep strongest instance per unique term, require min_doc_count
collocs_clean <- collocs_clean %>%
  group_by(term) %>%
  arrange(desc(n), desc(llr)) %>%
  slice(1) %>%
  ungroup() %>%
  filter(n >= min_doc_count)

logi("Collocation candidates: %d (doc_count >= %d).",
     nrow(collocs_clean), min_doc_count)

## =============================
## UNION & FINAL CLEANUP
## =============================

colloc_phrases <- collocs_clean %>%
  transmute(phrase = term, award_ids = award_ids, source = "collocation")

noun_phrases <- noun_phrase_counts %>%
  transmute(phrase = nounphrase, award_ids = award_ids, source = "nounphrase")

# Remove phrases that begin or end with stopwords; normalize spaces
stop_pattern <- paste0("\\b(", paste(sw, collapse = "|"), ")\\b")

clean_edges <- bind_rows(colloc_phrases, noun_phrases) %>%
  mutate(
    phrase = str_replace_all(phrase, "_", " "),
    phrase = str_squish(phrase)
  ) %>%
  filter(
    phrase != "",
    !str_detect(phrase, paste0("^", stop_pattern, "\\s")),
    !str_detect(phrase, paste0("\\s", stop_pattern, "$"))
  )

# Keep only multi-word phrases (at least one space)
clean_edges <- clean_edges %>% filter(str_detect(phrase, " "))

# Aggregate by phrase (union award_ids, join sources)
keyword_candidates <- clean_edges %>%
  group_by(phrase) %>%
  summarise(
    award_ids = list(sort(unique(unlist(award_ids)))),
    sources   = paste(sort(unique(source)), collapse = "+"),
    .groups   = "drop"
  ) %>%
  mutate(
    doc_count = lengths(award_ids),
    omit      = "",     # to be edited downstream (shiny)
    method    = "",
    thematic  = "",
    kid       = as.character(row_number())
  ) %>%
  arrange(desc(doc_count))

# Remove generic or non-informative terms before saving
remove_terms <- c(
  "research", "project", "award", "dissertation", 
  "provide", "result", "contribute", "scholar", "support"
)

pattern_remove <- paste(remove_terms, collapse = "|")

keyword_candidates <- keyword_candidates %>%
  filter(!str_detect(tolower(phrase), pattern_remove))

logi("Total unique phrases: %d", nrow(keyword_candidates))

## =============================
## WRITE OUTPUTS
## =============================

ensure_dir(out_dir)

# Convert award_ids list-column to a comma-separated string for CSV output
keyword_candidates_out <- keyword_candidates %>%
  mutate(
    award_ids = vapply(
      award_ids,
      function(ids) paste(ids, collapse = ","),
      FUN.VALUE = character(1)
    )
  )

logi("Writing: %s", out_file)
write_csv(keyword_candidates_out, out_file)

logi("Done.")

