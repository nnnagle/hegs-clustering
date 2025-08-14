#!/usr/bin/env python3
# ================================================================
# File: 02_sbert_cv_optuna.py
# Purpose: Run k-fold (folds 1–4) cross-validated hyperparameter
#          search (Optuna) for SBERT using keyword-pair similarity.
#
# Inputs (CSV):
#   - data/interim/kw_pairs_with_folds.csv
#       required columns:
#         * kw1, kw2  (strings)
#         * fold      (int; we will STRICTLY use folds 1..4)
#         * value     (float in [0,1]; preserved from kw_pairs.csv)
#
# Outputs:
#   - Console logs and Optuna study stats (best params, CV means/std)
#   - No artifacts are written by default (pipeline-friendly).
#
# Notes:
#   - Fold 5 is never read or evaluated (kept as pristine holdout).
#   - Model backbone: sentence-transformers/all-mpnet-base-v2
#   - CV metric: Spearman correlation on cosine similarity.
# ================================================================

import os
import sys
import json
import math
import time
from statistics import mean, pstdev

import numpy as np
import pandas as pd

# SBERT / Transformers / Torch
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, datasets
from transformers import AutoTokenizer
import torch
import optuna
from collections import Counter

# ---------------- Environment & Threads ----------------
# Keep tokenizer multi-processing quiet and control thread usage for reproducibility.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
# Make CUDA allocator less fragile (helps fragmentation on some GPUs).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TORCH_THREADS = 1  # cap intraop threads
try:
    torch.set_num_threads(int(TORCH_THREADS))
except Exception:
    pass

# ---------------- Utilities ----------------
def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def log(msg: str, *args):
    if args:
        msg = msg % args
    sys.stdout.write(f"[{timestamp()}] {msg}\n")
    sys.stdout.flush()

def ensure_file(path: str, purpose: str = "input"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {purpose}: {path}")

def check_required_columns(df: pd.DataFrame, req: set, where: str):
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{where}: missing required columns: {sorted(missing)}")

def assert_nonempty(df: pd.DataFrame, where: str):
    if df.empty:
        raise ValueError(f"{where}: no rows after filtering.")

# ================================================================
# ---------------------------- CONFIG ----------------------------
# ================================================================
IN_PAIRS_CSV = "data/interim/kw_pairs_with_folds.csv"   # from upstream R
USE_FOLDS = [1, 2, 3, 4]                                # strictly ignore fold 5
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_SEQ_LEN = 64
N_TRIALS = 30                                           # Optuna trial count
EVAL_SIM = evaluation.SimilarityFunction.COSINE

# Tokenizer threshold for diagnostics
LONG_PAIR_THRESH = 12  # total subword tokens across kw1+kw2

# ================================================================
# --------------------------- LOAD DATA --------------------------
# ================================================================
ensure_file(IN_PAIRS_CSV, "R output")
log("Reading pairs: %s", IN_PAIRS_CSV)
df_all = pd.read_csv(IN_PAIRS_CSV)

# Required schema checks
REQUIRED = {"kw1", "kw2", "fold"}
check_required_columns(df_all, REQUIRED, "kw_pairs_with_folds.csv")
if "value" not in df_all.columns:
    raise ValueError("Column 'value' not found; upstream pipeline should preserve it from kw_pairs.csv.")

# Basic cleaning & bounds
df_all = (
    df_all
    .dropna(subset=["kw1", "kw2", "value"])
    .loc[lambda d: (d["value"] >= 0.0) & (d["value"] <= 1.0)]
    .reset_index(drop=True)
)
df_all["fold"] = df_all["fold"].astype(int)

# Strictly keep folds 1..4 for CV
df_cv = df_all[df_all["fold"].isin(USE_FOLDS)].copy()
assert_nonempty(df_cv, "CV folds 1–4 subset")
log("Rows total: %d | CV rows (folds 1–4): %d", len(df_all), len(df_cv))

# ================================================================
# ---------------------- DIAGNOSTICS (PRE) -----------------------
# ================================================================
log("Running data-balance and tokenizer diagnostics...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def _ntoks(s: str) -> int:
    return len(tok.encode(str(s), add_special_tokens=False))

# 1) Size & label distribution per fold
size_label = df_cv.groupby("fold").agg(
    n=("value", "size"),
    label_mean=("value", "mean"),
    label_std=("value", "std"),
)
log("[Fold sizes & label moments]\n%s", size_label)

# 2) Binned label histograms (proportions)
bins = pd.IntervalIndex.from_tuples([(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0)], closed="right")
df_cv["value_bin"] = pd.cut(df_cv["value"], bins)
hist = pd.crosstab(df_cv["fold"], df_cv["value_bin"], normalize="index")
log("[Label distribution by fold]\n%s", hist)

# 3) Cross-fold leakage: same unordered pair across folds?
pairs_unordered = df_cv.apply(lambda r: tuple(sorted((str(r.kw1), str(r.kw2)))), axis=1)
dup_mask = pairs_unordered.duplicated(keep=False)
leak = (
    df_cv.loc[dup_mask]
    .assign(pair=pairs_unordered[dup_mask])
    .groupby(["pair", "fold"])
    .size()
    .unstack(fill_value=0)
)
leak_crossfold = leak[leak.gt(0).sum(axis=1) > 1]
if leak_crossfold.empty:
    log("[Potential cross-fold duplicates]\nNone")
else:
    log("[Potential cross-fold duplicates] showing head()\n%s", leak_crossfold.head())

# 4) Subword token length stats
for col in ["kw1", "kw2"]:
    df_cv[f"{col}_ntoks"] = df_cv[col].astype(str).apply(_ntoks)
df_cv["pair_ntoks"] = df_cv["kw1_ntoks"] + df_cv["kw2_ntoks"]
len_stats = df_cv.groupby("fold")[["kw1_ntoks", "kw2_ntoks", "pair_ntoks"]].describe()
log("[Subword token stats by fold]\n%s", len_stats)

long_share = (
    df_cv.assign(is_long=lambda d: d["pair_ntoks"] > LONG_PAIR_THRESH)
    .groupby("fold")["is_long"]
    .mean()
)
log("[Share of pairs with >%d subword tokens by fold]\n%s", LONG_PAIR_THRESH, long_share)

# 5) Rare-subword difficulty proxy
def _tokens_in_df(df: pd.DataFrame):
    for col in ["kw1", "kw2"]:
        for s in df[col].astype(str):
            yield from tok.encode(s, add_special_tokens=False)

def _mean_inverse_freq(val_df: pd.DataFrame, freq_counter: Counter) -> float:
    vals = []
    for col in ["kw1", "kw2"]:
        for s in val_df[col].astype(str):
            ids = tok.encode(s, add_special_tokens=False)
            vals.extend([1.0 / max(1, freq_counter[i]) for i in ids])
    return float(np.mean(vals)) if vals else 0.0

log("[Rare-subword proxy: mean inverse token frequency (per-fold validation)]")
for f in USE_FOLDS:
    train_df = df_cv[df_cv["fold"] != f]
    val_df = df_cv[df_cv["fold"] == f]
    freq = Counter(_tokens_in_df(train_df))
    mif = _mean_inverse_freq(val_df, freq)
    log("  Fold %d: %.6f", f, mif)

# ================================================================
# --------------------- BUILD CV OBJECTS -------------------------
# ================================================================
def make_input_examples(df_subset: pd.DataFrame):
    return [
        InputExample(texts=[str(a), str(b)], label=float(v))
        for a, b, v in df_subset[["kw1", "kw2", "value"]].itertuples(index=False, name=None)
    ]

fold_to_examples = {}
fold_to_eval = {}
for f in USE_FOLDS:
    val_df = df_cv[df_cv["fold"] == f]
    train_df = df_cv[df_cv["fold"] != f]
    if val_df.empty or train_df.empty:
        raise ValueError(f"Fold {f}: train or val split is empty. Check your data balance.")

    fold_to_examples[f] = make_input_examples(train_df)
    fold_to_eval[f] = evaluation.EmbeddingSimilarityEvaluator(
        sentences1=val_df["kw1"].astype(str).tolist(),
        sentences2=val_df["kw2"].astype(str).tolist(),
        scores=val_df["value"].astype(float).tolist(),
        main_similarity=EVAL_SIM,
    )

log("Prepared per-fold train/eval objects for folds: %s", USE_FOLDS)

# ================================================================
# --------------------- OPTUNA: K-FOLD CV ------------------------
# ================================================================
torch.cuda.empty_cache()

def objective(trial: optuna.Trial) -> float:
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    warm = trial.suggest_float("warmup_frac", 0.05, 0.20)
    epochs = trial.suggest_int("epochs", 2, 5)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    fold_scores = []
    for f in USE_FOLDS:
        # Fresh model per fold (no leakage)
        model = SentenceTransformer(MODEL_NAME)
        model.max_seq_length = MAX_SEQ_LEN

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

        # Evaluator returns a float (Spearman) or a dict depending on version.
        eval_out = fold_to_eval[f](model)
        if isinstance(eval_out, dict):
            spearman = eval_out.get("spearman_cosine") or eval_out.get("cosine_spearman")
            if spearman is None:
                raise RuntimeError("Evaluator did not return a Spearman score.")
        else:
            spearman = float(eval_out)

        fold_scores.append(spearman)

        # Free VRAM between folds (important on small GPUs)
        del model
        torch.cuda.empty_cache()

    # Record per-trial diagnostics
    trial.set_user_attr("fold_scores", fold_scores)
    trial.set_user_attr("cv_std", pstdev(fold_scores) if len(fold_scores) > 1 else 0.0)

    return mean(fold_scores)

log("Starting Optuna study (trials=%d)...", N_TRIALS)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

log("Best params: %s", study.best_params)
log("Best mean CV Spearman (folds 1–4): %.6f", study.best_value)
log("Best trial per-fold: %s", study.best_trial.user_attrs.get("fold_scores"))
log("Best trial CV std: %s", study.best_trial.user_attrs.get("cv_std"))

# Difficulty balance across all trials (is one fold consistently easiest/hardest?)
fold_mat = np.array([t.user_attrs["fold_scores"] for t in study.trials if "fold_scores" in t.user_attrs])
if fold_mat.size:
    per_fold_mean = fold_mat.mean(0)
    per_fold_std = fold_mat.std(0, ddof=0)
    log("[Per-fold difficulty across all trials]")
    for i, (m, s) in enumerate(zip(per_fold_mean, per_fold_std), start=1):
        log("  Fold %d: mean=%.4f, std=%.4f", i, m, s)
else:
    log("[Per-fold difficulty across trials] No recorded fold_scores beyond the best trial.")

# ------------------------------- END ------------------------------
# - This script never reads/evaluates fold 5.
# - For a final model: retrain on folds 1–4 using study.best_params,
#   still without touching fold 5 (hold it out for a true final check).
