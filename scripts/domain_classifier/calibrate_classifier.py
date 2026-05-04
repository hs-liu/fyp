# scripts/calibrate_classifier.py
import torch
import json
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import pandas as pd
import sys

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR  = "/vol/bitbucket/hl2622/fyp/data"
MODEL_DIR = "/vol/bitbucket/hl2622/fyp/models/domain_classifier"
HF_CACHE  = "/vol/bitbucket/hl2622/huggingface_cache"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + val data
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

df = pd.read_csv(f"{DATA_DIR}/classifier_training_data.csv")
val_df = df[df["split"] == "validation"].reset_index(drop=True)

# Collect logits on val set
from scripts.domain_classifier.train_domain_classifier import MedQADataset
val_ds = MedQADataset(val_df, tokenizer)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

all_logits, all_labels = [], []
with torch.no_grad():
    for batch in val_dl:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labs = batch["label"]
        logits = model(ids, attention_mask=mask).logits
        all_logits.append(logits.cpu())
        all_labels.append(labs)

all_logits = torch.cat(all_logits)
all_labels = torch.cat(all_labels)

# Learn temperature T via NLL minimisation
temperature = nn.Parameter(torch.ones(1))
optimizer   = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
criterion   = nn.CrossEntropyLoss()

def eval_t():
    optimizer.zero_grad()
    loss = criterion(all_logits / temperature, all_labels)
    loss.backward()
    return loss

optimizer.step(eval_t)
T = temperature.item()
print(f"Optimal temperature: {T:.4f}")

# Save
with open(f"{MODEL_DIR}/temperature.json", "w") as f:
    json.dump({"temperature": T}, f)
print(f"Saved → {MODEL_DIR}/temperature.json")