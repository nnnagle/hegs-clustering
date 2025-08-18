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
#     - Must contain columns: 'awaxrd_id' and 'abstract_clean'
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
  library(stringi)
  library(purrr)
  library(spacyr)     # spaCy noun phrases
  library(text2vec)   # collocations
  library(stopwords)  # stopword lists
  library(DBI)
  library(RSQLite)
})

## =============================
## PARAMETERS (edit here)
## =============================

# Paths
input_file   <- "data/interim/HEGS_clean_df.csv"
out_dir      <- "data/interim"
out_file    <- file.path(out_dir, "keyword_candidates.csv")
db_path <- "data/interim/keywords.sqlite"
db <- dbConnect(SQLite(), db_path)

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

source('R/proj_utilities.R')


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
abs_df <- read_csv(input_file, show_col_types = FALSE)

require_cols(abs_df, c("award_id", "abstract_clean"), "Abstracts")

abs_df <- abs_df %>%
  filter(!is.na(abstract_clean), nchar(abstract_clean) > 0)

n_docs <- nrow(abs_df)
logi("Loaded %d abstracts after filtering on 'is empty'.", n_docs)

if (n_docs == 0L) stop("No abstracts to process after filtering on 'is empty'.")

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
tokens_df <- spacy_tokenize(abs_df$abstract_clean, what = "word", output = "data.frame") %>%
  mutate(doc_index = as.integer(str_extract(doc_id, "\\d+")),
         award_id  = abs_df$award_id[doc_index]) %>%
  group_by(doc_id) %>%
  mutate(pos=row_number()) %>%
  ungroup()

spacy_phrases <- spacy_extract_nounphrases(abs_df$abstract_clean) %>%
  mutate(doc_index = as.integer(str_extract(doc_id, "\\d+")),
         award_id  = abs_df$award_id[doc_index],
         pre_id = start_id-1,
         post_id = start_id+length)
# Collapse hyphenated words
starts_with_hyphen <- which(str_detect(spacy_phrases$text, '^-'))
ends_with_hyphen <- which(str_detect(spacy_phrases$text, '-$'))
prefix <- left_join(spacy_phrases[starts_with_hyphen,], tokens_df, by=c('doc_index','award_id','pre_id'='pos')) %>%
  pull(token)
suffix <- left_join(spacy_phrases[ends_with_hyphen,], tokens_df, by=c('doc_index','award_id','post_id'='pos')) %>%
  pull(token)

spacy_phrases$text[starts_with_hyphen] <- paste0(prefix, spacy_phrases$text[starts_with_hyphen])
spacy_phrases$text[ends_with_hyphen] <- paste0(spacy_phrases$text[ends_with_hyphen], suffix)


# spacyr uses doc_id like "text1", "text2", ...
# Reconstruct mapping to award_id by position
spacy_phrases <- spacy_extract_nounphrases(abs_df$abstract_clean) %>%
  mutate(doc_index = as.integer(str_extract(doc_id, "\\d+")),
         award_id  = abs_df$award_id[doc_index])


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
    word_count = str_count(nounphrase, "[^\\s-]+")
  ) %>%
  filter(word_count>=2) %>%
  group_by(nounphrase, award_id) %>%
  summarise(
    n=n(),
    .groups = "drop"
  )  %>%
  group_by(nounphrase) %>%
  summarise(
    doc_count = n_distinct(award_id),
    award_id = list(award_id),   # vector of award_ids per nounphrase
    n        = list(n),          # matching vector of n per award_id
    .groups = "drop"
  ) %>%
  filter(doc_count>=min_doc_count)


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

tokens <- word_tokenizer(preprocessor(abs_df$abstract_clean))
it <- itoken(tokens, progressbar = FALSE)
tok_vocab <- create_vocabulary(it)

colloc_model <- Collocations$new(
  collocation_count_min = as.integer(collocation_count_min),
  pmi_min               = collocation_pmi_min
)
colloc_model$fit(it, n_iter = 2)
colloc_it <- colloc_model$transform(it)
colloc_vocab <- create_vocabulary(colloc_it)


filtered <- grepl("^(a|an|the|this|that|these|those|which|where)_",colloc_vocab$term )
colloc_vocab <- colloc_vocab[!filtered,] %>%
  filter(str_detect(term,'_'))

# Extract & clean collocation stats
collocs_clean <- colloc_vocab %>%
  mutate(term = str_replace_all(term, "_", " "))

# Trim leading/trailing stopwords from collocations
sw <- tolower(gsub("'", "", stopwords(language = stop_lang)))
sw <- c(sw, 'also', 'used','can','different', 'across', 'among', 'award', 'based', 'different','dissertation','doctoral',
        'help','important','many','may','much','new', 'order', 'processes','project','promising',
        'provide','relat','research','results', 's ', 'several', 'significant',
        'specific', 'study','theoretical','thereby','understanding','university', 'use ','used','using',
        'within')

collocs_clean$term <- vapply(collocs_clean$term, trim_term, stopwords_vec = sw, FUN.VALUE = character(1))
collocs_clean$term <- str_squish(collocs_clean$term)
collocs_clean <- collocs_clean %>%
  filter(str_detect(term, ' '))

# Build doc_texts for regex matching of terms back to documents
doc_texts <- vapply(tokens, paste, collapse = " ", FUN.VALUE = character(1))

# Map terms to award_ids via regex search over doc_texts
logi("Mapping collocations back to documents...")
collocs_clean <- as_tibble(collocs_clean)
collocs_clean <- collocs_clean %>%
  select(term) %>%
  distinct() %>%
  mutate(
    term_norm = str_squish(term),
    matches   = map(term, ~{
      cts <- stri_count_fixed(
        doc_texts, .x
      )
      nz <- which(cts > 0L)
      list(
        doc_count = length(nz),
        award_id = abs_df$award_id[nz],
        n    = unname(cts[nz])
      )
    })
  ) %>%
  tidyr::hoist(matches, doc_count = "doc_count", award_id = "award_id", n = "n") %>%
  filter(doc_count>=5)

logi("Collocation candidates: %d (doc_count >= %d).",
     nrow(collocs_clean), min_doc_count)

## =============================
## UNION & FINAL CLEANUP
## =============================

colloc_phrases <- collocs_clean %>%
  transmute(phrase = term, doc_count=doc_count, award_ids = award_id, n=n, source = "collocation")

noun_phrases <- noun_phrase_counts %>%
  transmute(phrase = nounphrase, doc_count=doc_count, award_ids = award_id, n=n, source = "nounphrase") %>%
  mutate(phrase = str_replace_all(phrase,'-',' '))

# Remove phrases that begin with stopwords; normalize spaces
stop_pattern <- pattern <- paste0("^(", paste(sw, collapse = "|"), ")")


# Reasonable defaults for speed + safety on a local file
quiet_dbExecute <- function(conn, sql) invisible(DBI::dbExecute(conn, sql))
quiet_dbExecute(db, "PRAGMA journal_mode=WAL;")
quiet_dbExecute(db, "PRAGMA synchronous=NORMAL;")

# Ensure table exists once (schema matches your CSV output)
quiet_dbExecute(db, "
CREATE TABLE IF NOT EXISTS keyword_candidates (
  kid TEXT PRIMARY KEY,
  phrase TEXT,
  award_ids TEXT,      -- comma string, same as CSV
  omit INTEGER CHECK (omit IN (0,1)),
  method INTEGER CHECK (method IN (0,1)),
  thematic INTEGER CHECK(thematic IN (0,1))
);
")
quiet_dbExecute(db, "CREATE INDEX IF NOT EXISTS idx_keyword_phrase ON keyword_candidates(phrase);")


keyword_candidates <- full_join(colloc_phrases, noun_phrases, by='phrase') %>%
  mutate(
    doc_count = if_else(!is.na(doc_count.x), doc_count.x, doc_count.y),
    award_ids = if_else(!is.na(doc_count.x), award_ids.x, award_ids.y),
    n = if_else(!is.na(doc_count.x), n.x, n.y),
    source = if_else(!is.na(doc_count.x), "collocation", "nounphrase"),
    source = if_else(!is.na(doc_count.x) & !(is.na(doc_count.y)), "collocation+nounphrase", source)
  ) %>%
  select(phrase, doc_count, award_ids, n, source) %>%
  filter(
    phrase != "",
    !str_detect(phrase, paste0("^", stop_pattern, "\\s"))  ) %>%
  arrange(phrase) %>%
  mutate(
    omit      = "",     # to be edited downstream (shiny)
    method    = "",
    thematic  = "",
    kid       = as.character(row_number())
  ) %>%
  arrange(desc(doc_count))



logi("Total unique phrases: %d", nrow(keyword_candidates))

## =============================
## WRITE OUTPUTS
## =============================

ensure_dir(out_dir)
# Convert award_ids list-column to a comma-separated string for CSV output
# 3) Persist updated candidates (award_ids back to comma string) — CSV + SQLite
keyword_candidates_out <- keyword_candidates %>%
  mutate(
    award_ids = vapply(award_ids, function(x) paste(x, collapse = ","), character(1)),
    omit      = ifelse(is.na(omit), FALSE, omit),
    method    = ifelse(is.na(method), FALSE, method),
    thematic  = ifelse(is.na(thematic), FALSE, thematic)
  ) %>%
  # ensure SQLite-friendly types
  mutate(
    omit     = as.integer(omit),
    method   = as.integer(method),
    thematic = as.integer(thematic)
  ) %>%
  select(kid, phrase, award_ids, omit, method, thematic)

# Write CSV (default quote='needed' is smaller & faster)
logi("Writing: %s", out_file)
readr::write_csv(keyword_candidates_out, out_file)

# Mirror into SQLite (single transactional replace)
logi("Writing: %s", db_path)
invisible(dbWithTransaction(db, {
  # overwrite table contents atomically
  dbExecute(db, "DELETE FROM keyword_candidates;")
  dbWriteTable(db, "keyword_candidates", keyword_candidates_out, append = TRUE)
}))


logi("Done.")

