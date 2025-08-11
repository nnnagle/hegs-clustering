import pandas as pd
import os
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, datasets

# 1) Make CUDA allocator less fragile (helps fragmentation)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# 1) Load & clean
df = pd.read_csv("train_pairs.csv").dropna(subset=["kw1","kw2","value"])
df = df[(df["value"] >= 0.0) & (df["value"] <= 1.0)].reset_index(drop=True)

val_df = pd.read_csv("test_pairs.csv").dropna(subset=["kw1","kw2","value"])
val_df = val_df[(val_df["value"] >= 0.0) & (val_df["value"] <= 1.0)].reset_index(drop=True)


# 2) Build InputExamples
train_examples = [
    InputExample(texts=[str(a), str(b)], label=float(v))
    for a,b,v in df[["kw1","kw2","value"]].itertuples(index=False, name=None)
]

# validation evaluator
val_evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sentences1 = val_df["kw1"].astype(str).tolist(),
    sentences2 = val_df["kw2"].astype(str).tolist(),
    scores     = val_df["value"].astype(float).tolist(),
    main_similarity = evaluation.SimilarityFunction.COSINE
)

# 3) Model + DataLoader + Loss
model = SentenceTransformer("all-mpnet-base-v2")  # or 'all-mpnet-base-v2'
train_dataloader = datasets.NoDuplicatesDataLoader(train_examples, batch_size=4)
train_loss = losses.CosineSimilarityLoss(model)   # labels in [0,1] are OK

# 4) Train
epochs = 3
warmup_steps = int(0.1 * len(train_dataloader) * epochs)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    evaluator=val_evaluator,
    evaluation_steps=max(50, len(train_dataloader)//4),
    output_path="finetuned_model",
    show_progress_bar=True,
    optimizer_params={"lr": 2e-5},
    use_amp=True
)

from itertools import product
import torch
torch.cuda.empty_cache()

import optuna
from sentence_transformers import SentenceTransformer, losses, datasets

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
        use_amp=True,                      # if GPU
        evaluation_steps=max(50, len(train_dl)//4),
        evaluator=val_evaluator
    )
    scores = val_evaluator(model)         # dict
    spearman = scores.get("spearman_cosine") or scores.get("cosine_spearman")
    return spearman

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print(study.best_params, study.best_value)














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
        evaluation_steps=max(50, len(train_dataloader)//4),
        show_progress_bar=False,
        optimizer_params={"lr": lr},
        use_amp=True
    )
    score = val_evaluator(model).get('spearman_cosine')  # Spearman
    results.append((epochs, lr, warmup_frac, score))
    print(f"\nEpochs: {epochs}, Learning Rate: {lr:.2e}, Warmup Fraction: {warmup_frac:.2f}, Validation Score (Spearman): {score:.4f}")





