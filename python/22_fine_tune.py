import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, datasets

# Make CUDA allocator less fragile (helps fragmentation)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -------------------------------------------------------------------
# 1) Load R output and restrict to folds 1..4 ONLY (strictly ignore 5)
# -------------------------------------------------------------------
pairs_path = "data/interim/kw_pairs_with_folds.csv"   # from the R script
if not os.path.isfile(pairs_path):
    raise FileNotFoundError(f"Expected R output at {pairs_path}")

df_all = pd.read_csv(pairs_path)

# Required columns
required = {"kw1", "kw2", "fold"}
missing = required - set(df_all.columns)
if missing:
    raise ValueError(f"Missing required columns in {pairs_path}: {missing}")
if "value" not in df_all.columns:
    raise ValueError("Column 'value' not found. The R pipeline should preserve it from kw_pairs.csv.")

# Clean
df_all = df_all.dropna(subset=["kw1", "kw2", "value"])
df_all = df_all[(df_all["value"] >= 0.0) & (df_all["value"] <= 1.0)].reset_index(drop=True)
df_all["fold"] = df_all["fold"].astype(int)

# Only folds 1..4 for CV; do not even keep fold 5 in memory
cv_folds = [1, 2, 3, 4]
df_cv = df_all[df_all["fold"].isin(cv_folds)].copy()
if df_cv.empty:
    raise ValueError("No rows found for folds 1–4 to run CV.")

# Build per-fold data once to avoid recomputing inside trials
def make_input_examples(df_subset: pd.DataFrame):
    return [
        InputExample(texts=[str(a), str(b)], label=float(v))
        for a, b, v in df_subset[["kw1", "kw2", "value"]].itertuples(index=False, name=None)
    ]

fold_to_examples = {}
fold_to_eval = {}
for f in cv_folds:
    val_df = df_cv[df_cv["fold"] == f]
    train_df = df_cv[df_cv["fold"] != f]
    if val_df.empty or train_df.empty:
        raise ValueError(f"Fold {f}: train or val split is empty. Check your data balance.")

    fold_to_examples[f] = make_input_examples(train_df)
    fold_to_eval[f] = evaluation.EmbeddingSimilarityEvaluator(
        sentences1=val_df["kw1"].astype(str).tolist(),
        sentences2=val_df["kw2"].astype(str).tolist(),
        scores=val_df["value"].astype(float).tolist(),
        main_similarity=evaluation.SimilarityFunction.COSINE,
    )

# -------------------------------------------------------------------
# 2) Optuna: K-fold CV on folds 1..4
# -------------------------------------------------------------------
import torch
torch.cuda.empty_cache()

import optuna
from statistics import mean
from sentence_transformers import SentenceTransformer, losses, datasets  # safe re-imports

def objective(trial):
    # Hyperparams to tune
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    warm = trial.suggest_float("warmup_frac", 0.05, 0.20)
    epochs = trial.suggest_int("epochs", 2, 5)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    fold_scores = []
    for f in cv_folds:
        # Fresh model per fold (no leakage)
        model = SentenceTransformer("all-mpnet-base-v2")
        model.max_seq_length = 64

        train_loss = losses.CosineSimilarityLoss(model)
        train_dl = datasets.NoDuplicatesDataLoader(fold_to_examples[f], batch_size=batch_size)

        # Warmup proportional to total steps for this fold
        warmup_steps = int(warm * len(train_dl) * epochs)

        model.fit(
            train_objectives=[(train_dl, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=False,
            optimizer_params={"lr": lr},
            use_amp=True,
            evaluation_steps=max(50, len(train_dl) // 4),
            evaluator=fold_to_eval[f],
        )

        scores = fold_to_eval[f](model)  # dict
        spearman = scores.get("spearman_cosine") or scores.get("cosine_spearman")
        if spearman is None:
            raise RuntimeError("Evaluator did not return a Spearman score.")
        fold_scores.append(spearman)

        # Free VRAM between folds (especially important on small GPUs)
        del model
        torch.cuda.empty_cache()

    # Return mean CV score across folds 1..4
    return mean(fold_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best params:", study.best_params)
print("Best mean CV Spearman (folds 1–4):", study.best_value)

# NOTE:
# - This script *never* reads/evaluates fold 5.
# - If you later want a final model, retrain on folds 1–4 with study.best_params,
#   still without touching fold 5. (Keep fold 5 as a pristine holdout if desired.)
