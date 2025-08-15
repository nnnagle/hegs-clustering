# NSF/HEGS Clustering

Cluster HEGS portfolio with fine-tuned sentence transformers

## Interesting features
- Shiny app with multiple samplers to label keyword pairs
- Data splits by sampling edges of keyword graph
- Hyperparameter selection with optuna

## Contents
- `R/05_HEGS_clean.R` Cleain the HEGS abstracts
- `R/10_keyword_candidates.R` Create keword candidates from noun_phrases and collocations
- `R/15_sentence_transformer.R` Create embeddings
- `R/app.R` Shiny labeling app for keyword pairs + metadata
- `R/20_train_split.R` Create training splits of keyword pairs using graph to avoid leakage
- `python/22_fine_tune.py` fine tuning on keyword pairs with CosineSimilarities
- `python/25_validate_pairs.py`
- `python/30_encode_corpus.py`
- `python/35_cluster.py`

## Data
- `data/raw/HEGS_awards.csv` 
- `data/interim/kw_pairs.csv` Created in 15_. Interactively updated by shiny app
