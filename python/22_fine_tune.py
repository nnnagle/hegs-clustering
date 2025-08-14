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

# --------------------------
# Diagnostics (pre-training)
# --------------------------
import numpy as np
from collections import Counter
from transformers import AutoTokenizer

# Use the same tokenizer family as the model
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

def _ntoks(s: str) -> int:
    return len(tok.encode(str(s), add_special_tokens=False))

print("\n=== DATA BALANCE DIAGNOSTICS ===")

# 1) Size and label distribution per fold
size_label = df_cv.groupby("fold").agg(
    n=("value", "size"),
    label_mean=("value", "mean"),
    label_std=("value", "std"),
)
print("\n[Fold sizes & label moments]\n", size_label)

# 2) Binned label histograms (proportions)
bins = pd.IntervalIndex.from_tuples([(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0)], closed="right")
df_cv["value_bin"] = pd.cut(df_cv["value"], bins)
hist = pd.crosstab(df_cv["fold"], df_cv["value_bin"], normalize="index")
print("\n[Label distribution by fold]\n", hist)

# 3) Cross-fold leakage: same unordered pair across folds?
pairs_unordered = df_cv.apply(lambda r: tuple(sorted((str(r.kw1), str(r.kw2)))), axis=1)
dup_mask = pairs_unordered.duplicated(keep=False)
leak = df_cv.loc[dup_mask].assign(pair=pairs_unordered[dup_mask]).groupby(["pair","fold"]).size().unstack(fill_value=0)
leak_crossfold = leak[leak.gt(0).sum(axis=1) > 1]
print("\n[Potential cross-fold duplicates]\n", leak_crossfold.head() if not leak_crossfold.empty else "None")

print("\n=== TOKENIZER-LEVEL DIAGNOSTICS ===")

# 4) Subword token length stats
for col in ["kw1", "kw2"]:
    df_cv[f"{col}_ntoks"] = df_cv[col].astype(str).apply(_ntoks)
df_cv["pair_ntoks"] = df_cv["kw1_ntoks"] + df_cv["kw2_ntoks"]

len_stats = df_cv.groupby("fold")[["kw1_ntoks","kw2_ntoks","pair_ntoks"]].describe()
print("\n[Subword token stats by fold]\n", len_stats)

thresh = 12  # tweak as needed
long_share = df_cv.assign(is_long=lambda d: d["pair_ntoks"] > thresh) \
                  .groupby("fold")["is_long"].mean()
print(f"\n[Share of pairs with >{thresh} subword tokens by fold]\n", long_share)

# 5) Rare-subword difficulty proxy: mean inverse token frequency in each fold's VAL set,
#    with frequencies computed from that fold's TRAIN set (lower == easier).
def _tokens_in_df(df):
    for col in ["kw1","kw2"]:
        for s in df[col].astype(str):
            yield from tok.encode(s, add_special_tokens=False)

def _mean_inverse_freq(val_df, freq_counter: Counter) -> float:
    vals = []
    for col in ["kw1","kw2"]:
        for s in val_df[col].astype(str):
            ids = tok.encode(s, add_special_tokens=False)
            vals.extend([1.0 / max(1, freq_counter[i]) for i in ids])
    return float(np.mean(vals)) if vals else 0.0

print("\n[Rare-subword proxy: mean inverse token frequency (per-fold validation)]")
for f in cv_folds:
    train_df = df_cv[df_cv["fold"] != f]
    val_df   = df_cv[df_cv["fold"] == f]
    freq = Counter(_tokens_in_df(train_df))
    mif = _mean_inverse_freq(val_df, freq)
    print(f"  Fold {f}: {mif:.6f}")

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
from statistics import mean, pstdev
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

        # Evaluator may return a float (Spearman) or a dict depending on version.
        eval_out = fold_to_eval[f](model)
        if isinstance(eval_out, dict):
            spearman = eval_out.get("spearman_cosine") or eval_out.get("cosine_spearman")
            if spearman is None:
                raise RuntimeError("Evaluator did not return a Spearman score.")
        else:
            spearman = float(eval_out)

        fold_scores.append(spearman)

        # Free VRAM between folds (especially important on small GPUs)
        del model
        torch.cuda.empty_cache()

    # Attach per-fold stats to the trial for later inspection
    trial.set_user_attr("fold_scores", fold_scores)
    trial.set_user_attr("cv_std", pstdev(fold_scores) if len(fold_scores) > 1 else 0.0)

    # Return mean CV score across folds 1..4
    return mean(fold_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best params:", study.best_params)
print("Best mean CV Spearman (folds 1–4):", study.best_value)
print("Best trial per-fold:", study.best_trial.user_attrs.get("fold_scores"))
print("Best trial CV std:", study.best_trial.user_attrs.get("cv_std"))

# Difficulty balance across all trials (is one fold consistently easiest/hardest?)
fold_mat = np.array([t.user_attrs["fold_scores"] for t in study.trials if "fold_scores" in t.user_attrs])
if fold_mat.size:
    per_fold_mean = fold_mat.mean(0)
    per_fold_std  = fold_mat.std(0, ddof=0)
    print("\n[Per-fold difficulty across all trials]")
    for i, (m, s) in enumerate(zip(per_fold_mean, per_fold_std), start=1):
        print(f"  Fold {i}: mean={m:.4f}, std={s:.4f}")
else:
    print("\n[Per-fold difficulty across trials] No recorded fold_scores beyond the best trial.")

# NOTE:
# - This script *never* reads/evaluates fold 5.
# - If you later want a final model, retrain on folds 1–4 with study.best_params,
#   still without touching fold 5. (Keep fold 5 as a pristine holdout if desired.)
