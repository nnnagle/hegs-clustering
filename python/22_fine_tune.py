import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, datasets

# Make CUDA allocator less fragile (helps fragmentation)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -------------------------------------------------------------------
# 1) Load R output and split by folds (train=1..4, val=5)
# -------------------------------------------------------------------
pairs_path = "data/interim/kw_pairs_with_folds.csv"   # from the R script
if not os.path.isfile(pairs_path):
    raise FileNotFoundError(f"Expected R output at {pairs_path}")

df_all = pd.read_csv(pairs_path)

# Basic column checks
required = {"kw1", "kw2", "fold"}
missing = required - set(df_all.columns)
if missing:
    raise ValueError(f"Missing required columns in {pairs_path}: {missing}")

# If your original data has a continuous label (e.g., similarity) keep it;
# otherwise, fail fast so you don't silently train on garbage.
if "value" not in df_all.columns:
    raise ValueError("Column 'value' not found. The R pipeline should preserve it from kw_pairs.csv.")

# Clean bounds and NAs
df_all = df_all.dropna(subset=["kw1", "kw2", "value"])
df_all = df_all[(df_all["value"] >= 0.0) & (df_all["value"] <= 1.0)].reset_index(drop=True)

# Enforce integer folds and filter to the ones we need
df_all["fold"] = df_all["fold"].astype(int)

train_df = df_all[df_all["fold"].isin([1, 2, 3, 4])].copy()
val_df   = df_all[df_all["fold"] == 5].copy()

if train_df.empty:
    raise ValueError("No training rows found for folds 1–4.")
if val_df.empty:
    raise ValueError("No validation rows found for fold 5.")

# -------------------------------------------------------------------
# 2) Build InputExamples and evaluator
# -------------------------------------------------------------------
train_examples = [
    InputExample(texts=[str(a), str(b)], label=float(v))
    for a, b, v in train_df[["kw1", "kw2", "value"]].itertuples(index=False, name=None)
]

val_evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sentences1=val_df["kw1"].astype(str).tolist(),
    sentences2=val_df["kw2"].astype(str).tolist(),
    scores=val_df["value"].astype(float).tolist(),
    main_similarity=evaluation.SimilarityFunction.COSINE,
)

# -------------------------------------------------------------------
# 3) Model + DataLoader + Loss
# -------------------------------------------------------------------
model = SentenceTransformer("all-mpnet-base-v2")
train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=4)
train_loss = losses.CosineSimilarityLoss(model)

# -------------------------------------------------------------------
# 4) Train (baseline run)
# -------------------------------------------------------------------
epochs = 3
warmup_steps = int(0.1 * len(train_dataloader) * epochs)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    evaluator=val_evaluator,
    evaluation_steps=max(50, len(train_dataloader) // 4),
    output_path="finetuned_model",
    show_progress_bar=True,
    optimizer_params={"lr": 2e-5},
    use_amp=True,
)

# -------------------------------------------------------------------
# 5) Optuna hyperparam search (still evaluates on fold 5)
# -------------------------------------------------------------------
from itertools import product
import torch
torch.cuda.empty_cache()

import optuna
from sentence_transformers import SentenceTransformer, losses, datasets  # re-import safe

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    warm = trial.suggest_float("warmup_frac", 0.05, 0.20)
    epochs = trial.suggest_int("epochs", 2, 5)

    model = SentenceTransformer("all-mpnet-base-v2")
    model.max_seq_length = 64
    train_loss = losses.CosineSimilarityLoss(model)
    train_dl = datasets.NoDuplicatesDataLoader(train_examples, batch_size=16)
    warmup_steps = int(warm * len(train_dl) * epochs)

    model.fit(
        train_objectives=[(train_dl, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=False,
        optimizer_params={"lr": lr},
        use_amp=True,
        evaluation_steps=max(50, len(train_dl) // 4),
        evaluator=val_evaluator,
    )

    scores = val_evaluator(model)
    spearman = scores.get("spearman_cosine") or scores.get("cosine_spearman")
    return spearman

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print(study.best_params, study.best_value)

# -------------------------------------------------------------------
# 6) Simple grid (optional) – still train on folds 1–4, validate on 5
# -------------------------------------------------------------------
epochs_list = [2, 3, 4]
lr_list = [1e-5, 2e-5, 5e-5]
warmup_frac_list = [0.05, 0.1, 0.2]

results = []
for epochs, lr, warmup_frac in product(epochs_list, lr_list, warmup_frac_list):
    warmup_steps = int(warmup_frac * len(train_dataloader) * epochs)
    model = SentenceTransformer("all-mpnet-base-v2")
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluator=val_evaluator,
        evaluation_steps=max(50, len(train_dataloader) // 4),
        show_progress_bar=False,
        optimizer_params={"lr": lr},
        use_amp=True,
    )
    score = val_evaluator(model).get("spearman_cosine")
    results.append((epochs, lr, warmup_frac, score))
    print(
        f"\nEpochs: {epochs}, Learning Rate: {lr:.2e}, Warmup Fraction: {warmup_frac:.2f}, "
        f"Validation Score (Spearman): {score:.4f}"
    )
