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
# Outputs (written under /model):
#   - /model/sbert_final_folds1-4/           (SentenceTransformer save dir)
#   - /model/best_params.json                (Optuna best hyperparameters)
#   - /model/holdout_metrics.json            (Spearman, Pearson, MAE, RMSE, Bias)
#   - /model/holdout_predictions.csv         (kw1, kw2, value, pred, error)
#   - /model/holdout_deciles.csv             (calibration by true-label deciles)
#   - Console logs and CV difficulty summary
#
# Notes:
#   - CV metric: Spearman correlation on cosine similarity.
#   - Fold 5 is kept pristine for final evaluation only.
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
# For stricter CUDA determinism in some GEMM paths
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(path):
        raise RuntimeError(f"Unable to create directory: {path}")

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
USE_FOLDS = [1, 2, 3, 4]                                # strictly ignore fold 5 during CV
HOLDOUT_FOLD = 5
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_SEQ_LEN = 64
N_TRIALS = 30                                           # Optuna trial count
EVAL_SIM = evaluation.SimilarityFunction.COSINE

SEED = 42                                               # <<< reproducibility seed
MODEL_DIR = "models"                                    # <<< output root for artifacts
FINAL_MODEL_SUBDIR = "sbert_final_folds1-4"
CHECKPOINTS_DIR = "models/checkpoints"

# Tokenizer threshold for diagnostics
LONG_PAIR_THRESH = 12  # total subword tokens across kw1+kw2

# ================================================================
# ------------------------ Reproducibility -----------------------
# ================================================================
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Deterministic kernels (slower but steadier)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Avoid TF32 drift on Ampere+
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# Be strict where possible; warn if an op lacks a deterministic impl
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

log("Reproducibility seed: %d", SEED)

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
ensure_dir(CHECKPOINTS_DIR)
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

        # Seeded DataLoader for stable shuffling (fallback if generator unsupported)
        g = torch.Generator().manual_seed(SEED + f)
        try:
            train_dl = datasets.NoDuplicatesDataLoader(fold_to_examples[f], batch_size=batch_size, generator=g)
        except TypeError:
            train_dl = datasets.NoDuplicatesDataLoader(fold_to_examples[f], batch_size=batch_size)

        # Warmup proportional to total steps for this fold
        warmup_steps = int(warm * len(train_dl) * epochs)

        model.fit(
            train_objectives=[(train_dl, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=False,
            optimizer_params={"lr": lr},
            use_amp=True,                 # flip to False for stricter determinism
            evaluation_steps=None,        # epoch-end only
            evaluator=fold_to_eval[f],
            checkpoint_path=CHECKPOINTS_DIR,            # ← disable
            checkpoint_save_steps=2000,      # ← disable
            checkpoint_save_total_limit=3 # ← disable
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

        # Free VRAM between folds
        del model
        torch.cuda.empty_cache()

    # Record per-trial diagnostics
    trial.set_user_attr("fold_scores", fold_scores)
    trial.set_user_attr("cv_std", pstdev(fold_scores) if len(fold_scores) > 1 else 0.0)

    return mean(fold_scores)

# Seeded sampler so trial sequence is reproducible
sampler = optuna.samplers.TPESampler(seed=SEED)
log("Starting Optuna study (trials=%d, seed=%d)...", N_TRIALS, SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS)

log("Best params: %s", study.best_params)
log("Best mean CV Spearman (folds 1–4): %.6f", study.best_value)
log("Best trial per-fold: %s", study.best_trial.user_attrs.get("fold_scores"))
log("Best trial CV std: %s", study.best_trial.user_attrs.get("cv_std"))

# Difficulty balance across all trials (is one fold consistently easiest/hardest?)
fold_mat = np.array([t.user_attrs["fold_scores"] for t in study.trials if "fold_scores" in t.user_attrs])
if fold_mat.size:
    per_fold_mean = fold_mat.mean(0)
    per_fold_std  = fold_mat.std(0, ddof=0)
    log("[Per-fold difficulty across all trials]")
    for i, (m, s) in enumerate(zip(per_fold_mean, per_fold_std), start=1):
        log("  Fold %d: mean=%.4f, std=%.4f", i, m, s)
else:
    log("[Per-fold difficulty across trials] No recorded fold_scores beyond the best trial.")

# Persist best params immediately
ensure_dir(MODEL_DIR)
best_params_path = os.path.join(MODEL_DIR, "best_params.json")
with open(best_params_path, "w") as f:
    json.dump(study.best_params, f, indent=2)
log("Wrote best params: %s", best_params_path)

# ================================================================
# ------ FINAL TRAIN (FOLDS 1–4) & HOLDOUT (FOLD 5) EVALUATION ---
# ================================================================
df_hold = df_all[df_all["fold"] == HOLDOUT_FOLD].copy()
assert_nonempty(df_hold, f"Holdout fold {HOLDOUT_FOLD}")

log("Final training on folds %s with best params, then evaluating on holdout fold %d.",
    USE_FOLDS, HOLDOUT_FOLD)

best = study.best_params
lr      = float(best["lr"])
warm    = float(best["warmup_frac"])
epochs  = int(best["epochs"])
bs      = int(best["batch_size"])

# Build full training set from folds 1–4
final_examples = make_input_examples(df_cv)

g_final = torch.Generator().manual_seed(SEED + 999)
try:
    final_dl = datasets.NoDuplicatesDataLoader(final_examples, batch_size=bs, generator=g_final)
except TypeError:
    final_dl = datasets.NoDuplicatesDataLoader(final_examples, batch_size=bs)

# Fresh model
final_model = SentenceTransformer(MODEL_NAME)
final_model.max_seq_length = MAX_SEQ_LEN

final_train_loss = losses.CosineSimilarityLoss(final_model)
final_warmup_steps = int(warm * len(final_dl) * epochs)

log("Final fit: epochs=%d, batch_size=%d, lr=%.2e, warmup_frac=%.2f (steps=%d).",
    epochs, bs, lr, warm, final_warmup_steps)

final_model.fit(
    train_objectives=[(final_dl, final_train_loss)],
    epochs=epochs,
    warmup_steps=final_warmup_steps,
    show_progress_bar=False,
    optimizer_params={"lr": lr},
    use_amp=True,             # set False for stricter determinism
    evaluation_steps=None,    # no mid-epoch noise
    evaluator=None,           # evaluate after training
    checkpoint_path=CHECKPOINTS_DIR,            # ← disable
    checkpoint_save_steps=2000,      # ← disable
    checkpoint_save_total_limit=3 # ← disable
)

# --- Save final model ---
final_model_dir = os.path.join(MODEL_DIR, FINAL_MODEL_SUBDIR)
ensure_dir(final_model_dir)
final_model.save(final_model_dir)
log("Saved final SentenceTransformer to: %s", final_model_dir)

# --- Holdout predictions (cosine similarity) ---
kw1_list = df_hold["kw1"].astype(str).tolist()
kw2_list = df_hold["kw2"].astype(str).tolist()
y_true   = df_hold["value"].astype(float).to_numpy()

# Encode with normalization so dot == cosine
emb1 = final_model.encode(kw1_list, batch_size=max(64, bs), convert_to_numpy=True, normalize_embeddings=True)
emb2 = final_model.encode(kw2_list, batch_size=max(64, bs), convert_to_numpy=True, normalize_embeddings=True)
y_pred = np.sum(emb1 * emb2, axis=1).astype(float)

# --- Error statistics ---
def _spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method="average")
    rb = pd.Series(b).rank(method="average")
    c = np.corrcoef(ra, rb)
    return float(c[0, 1]) if c.shape == (2, 2) else float("nan")

def _pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    c = np.corrcoef(a, b)
    return float(c[0, 1]) if c.shape == (2, 2) else float("nan")

err   = y_pred - y_true
mae   = float(np.mean(np.abs(err)))
rmse  = float(np.sqrt(np.mean(err ** 2)))
bias  = float(np.mean(err))
spr   = _spearman_np(y_pred, y_true)
pr    = _pearson_np(y_pred, y_true)

log("[HOLDOUT fold %d] n=%d  Spearman=%.6f  Pearson=%.6f  MAE=%.6f  RMSE=%.6f  Bias=%.6f",
    HOLDOUT_FOLD, len(y_true), spr, pr, mae, rmse, bias)

# --- Save metrics & predictions ---
metrics = {
    "seed": SEED,
    "model_name": MODEL_NAME,
    "max_seq_len": MAX_SEQ_LEN,
    "best_params": study.best_params,
    "cv_best_mean_spearman": study.best_value,
    "holdout_fold": HOLDOUT_FOLD,
    "n_holdout": int(len(y_true)),
    "spearman": spr,
    "pearson": pr,
    "mae": mae,
    "rmse": rmse,
    "bias": bias,
    "timestamp": timestamp(),
}
metrics_path = os.path.join(MODEL_DIR, "holdout_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
log("Wrote holdout metrics: %s", metrics_path)

pred_df = pd.DataFrame({
    "kw1": kw1_list,
    "kw2": kw2_list,
    "value": y_true,
    "pred": y_pred,
    "error": err,
})
pred_path = os.path.join(MODEL_DIR, "holdout_predictions.csv")
pred_df.to_csv(pred_path, index=False)
log("Wrote holdout predictions: %s", pred_path)

# Optional: quick calibration glance (bin by true label) + save
try:
    calib = pred_df.copy()
    calib["bin"] = pd.qcut(calib["value"], q=10, duplicates="drop")
    by_bin = calib.groupby("bin").agg(
        n=("value", "size"),
        true_mean=("value", "mean"),
        pred_mean=("pred", "mean"),
        mae=("pred", lambda s: float(np.mean(np.abs(s - calib.loc[s.index, "value"])))),
    ).reset_index()
    deciles_path = os.path.join(MODEL_DIR, "holdout_deciles.csv")
    by_bin.to_csv(deciles_path, index=False)
    log("Wrote holdout deciles table: %s", deciles_path)
except Exception:
    log("Skipped decile calibration table (insufficient variety or binning error).")

# Clean up GPU memory
del final_model
torch.cuda.empty_cache()

# ------------------------------- END ------------------------------
# - CV search uses folds 1–4 only.
# - Final model is trained on folds 1–4 with best params.
# - Holdout metrics reported on fold 5 and artifacts written under /model.
