import os
import re
import json
import time
import torch
import datasets
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sys
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp/src")
sys.path.insert(0, "/vol/bitbucket/hl2622/fyp/MedRAG/src")
from src.scripts.baselines.baseline_utils import format_question, evaluate_model


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
N_TEST = 500 
RESULTS_DIR = "./results/appendix"
os.makedirs(RESULTS_DIR, exist_ok=True)


print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

print("\nLoading dataset...")
dataset = datasets.load_dataset("bigbio/med_qa", "med_qa_en_source", trust_remote_code=True)
test_ds = list(dataset["test"])[:N_TEST]

BIOMISTRAL_PATH = "/vol/bitbucket/hl2622/fyp/src/models/biomistral-7b"
bm_tokenizer = AutoTokenizer.from_pretrained(BIOMISTRAL_PATH)
bm_model = AutoModelForCausalLM.from_pretrained(
    BIOMISTRAL_PATH,
    device_map="cuda:0",
    dtype=torch.float16,
)

def biomistral_fn(sample):
    prompt = format_question(sample)
    inputs = bm_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2000
    ).to(bm_model.device)

    # Remove trailing EOS token if present
    input_ids = inputs["input_ids"]
    if input_ids[0, -1] == bm_tokenizer.eos_token_id:
        input_ids = input_ids[:, :-1]
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = bm_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=bm_tokenizer.eos_token_id,
            eos_token_id=None
        )
    # Slice at the token level, not character level
    new_token_ids = output_ids[0][input_ids.shape[1]:]
    return bm_tokenizer.decode(new_token_ids, skip_special_tokens=True)

evaluate_model(
    biomistral_fn,
    test_ds,
    model_name="BioMistral-7B",
    save_path=f"{RESULTS_DIR}/results_biomistral_raw.csv",
    summary_path=f"{RESULTS_DIR}/more_test_summary.txt"
)