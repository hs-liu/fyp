import os
import sys
import json
import shutil
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR  = "/vol/bitbucket/hl2622/fyp/data"
MODEL_DIR = "/vol/bitbucket/hl2622/fyp/models/domain_classifier"
HF_CACHE  = "/vol/bitbucket/hl2622/huggingface_cache"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load label map from data builder ─────────────────────
print("Loading label map...")
label_map    = json.load(open(f"{DATA_DIR}/classifier_label_map.json"))
label2id     = label_map["label2id"]
id2label     = {int(k): v for k, v in label_map["id2label"].items()}
GROUP_LABELS = label_map["groups"]
NUM_CLASSES  = len(GROUP_LABELS)
print(f"  {NUM_CLASSES} classes: {GROUP_LABELS}")

# ── Load data ─────────────────────────────────────────────
print("\nLoading training data...")
df = pd.read_csv(f"{DATA_DIR}/classifier_training_data.csv")
print(f"  {len(df):,} samples")
print(df["label"].value_counts().to_string())

train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df   = df[df["split"] == "validation"].reset_index(drop=True)
test_df  = df[df["split"] == "test"].reset_index(drop=True)
print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# ── Dataset ───────────────────────────────────────────────
class MedQADataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts   = df["text"].tolist()
        self.labels  = df["label_id"].tolist()
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── Model ─────────────────────────────────────────────────
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    cache_dir=HF_CACHE,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
model = model.to(device)

train_ds = MedQADataset(train_df, tokenizer)
val_ds   = MedQADataset(val_df,   tokenizer)
test_ds  = MedQADataset(test_df,  tokenizer)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2)
test_dl  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=2)

# ── Class weights for imbalanced data ────────────────────
# Replace the class weights block with this
print("\nComputing class weights...")
train_labels = train_df["label_id"].tolist()

# Count samples per class
from collections import Counter
label_counts = Counter(train_labels)
total = len(train_labels)

# Soft weighting — square root instead of inverse frequency
# This balances without overcorrecting
weights = []
for i in range(NUM_CLASSES):
    count = label_counts.get(i, 1)
    weight = (total / (NUM_CLASSES * count)) ** 0.5  # sqrt dampens extreme weights
    weights.append(weight)

weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
print(f"  Soft class weights:")
for g, w in zip(GROUP_LABELS, weights):
    count = label_counts.get(GROUP_LABELS.index(g), 0)
    print(f"    {g:<35} n={count:<6} weight={w:.3f}")
# ── Training ──────────────────────────────────────────────
EPOCHS      = 3
optimizer   = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_dl) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps,
)

def evaluate(model, dl):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in dl:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labs   = batch["label"].to(device)
            logits = model(ids, attention_mask=mask).logits
            probs  = torch.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labs.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    acc = sum(p==l for p,l in zip(all_preds,all_labels)) / len(all_labels)
    return acc, all_preds, all_labels, all_probs

print(f"\nTraining for {EPOCHS} epochs...")
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dl):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labs = batch["label"].to(device)

        logits = model(ids, attention_mask=mask).logits
        loss   = criterion(logits, labs)   # weighted loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        if (step+1) % 50 == 0:
            print(f"  Epoch {epoch+1} step {step+1}/{len(train_dl)} "
                  f"loss: {total_loss/(step+1):.4f}")
            sys.stdout.flush()

    val_acc, _, _, _ = evaluate(model, val_dl)
    print(f"  Epoch {epoch+1} val accuracy: {val_acc:.4f}")
    sys.stdout.flush()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print(f"  ✅ Saved best model (val_acc={val_acc:.4f})")

# ── Test evaluation ───────────────────────────────────────
print("\nLoading best model for test evaluation...")
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
test_acc, preds, labels, probs = evaluate(model, test_dl)

print(f"\nTest accuracy: {test_acc:.4f}")
print("\nClassification report:")
print(classification_report(
    labels, preds,
    target_names=GROUP_LABELS,
    labels=list(range(NUM_CLASSES)),
    zero_division=0,
))

# ── Save label map to model dir ───────────────────────────
shutil.copy(f"{DATA_DIR}/classifier_label_map.json", f"{MODEL_DIR}/label_map.json")
print(f"\nModel saved → {MODEL_DIR}")
print("Done!")