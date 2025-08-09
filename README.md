# NSF/HEGS Clustering

Cluster HEGS portfolio with fine-tuned sentence transformers

## Contents
- `R/05_HEGS_clean.R` Cleain the HEGS abstracts
- `R/10_keyword_candidates.R` Create keword candidates from noun_phrases and ocllocations
- `R/15_sentence_transformer.R` Create embeddings
- `R/16_train_split.R` Create training splits of keyword pairs using graph to avoid leakage
- `R/20_tune_transformer.R` R portion of transformer tuning
- `R/app.R` Shiny labeling app for keyword pairs + metadataA
- `python/20_tune.py` fine tuning on keyword pairs with CosineSimilarities
- `python/25_validate_pairs.py`
- `python/30_encode_corpus.py`
- `python/35_cluster.py`

## Data (not committed)
- `data/raw/HEGS_awards.csv` 
- 
