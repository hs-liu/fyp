#!/usr/bin/env python
# coding: utf-8

# ## Baseline Evaluation
# This is to benchmarking baseline performance. 
# This should include following sections:
# 1. off-shelf LLMs on MedQA 
# 

# In[1]:


import torch 
import transformers
import datasets
import json 
import pandas as pd
import os 
from openai import OpenAI
from dotenv import load_dotenv


# In[2]:


dataset = datasets.load_dataset("bigbio/med_qa", "med_qa_en_source", trust_remote_code=True)
print(dataset)
print(dataset["train"][0])


# In[3]:


# format dataset questons 
def format_question(sample):
    print(sample)
    options = "\n".join(
        [f"{key}.{val}" for key, val in sample["options"]]
    )

    prompts = (
        f"You are a medical student. Answer the following question:\n\n"
        f"Question: {sample['question']}\n\n"
        f"Options:\n{options}\n\n"
        f"Answer with only the letter of the correct option (A, B, C, or D)."
    )

    return prompts 

# parse answer 
def parse_answer(ans):
    parsed_ans = ans.strip().upper()
    if parsed_ans in ["A", "B", "C", "D"]:
        return parsed_ans
    return "UNKNOWN"

# evaluate model 
def evaluate_model(model_fn, questions, model_name="model"):
    total = len(questions)
    correct = 0 
    res = []

    for i, sample in enumerate(questions):
        prompt = format_question(sample)
        model_answer = model_fn(prompt)
        parsed_answer = parse_answer(model_answer)
        ground_truth = sample["answer_idx"]
        is_correct = parsed_answer == ground_truth
        if is_correct:
            correct += 1

        results.append({
            "id": i, 
            "question": sample["question"],
            "ground_truth": ground_truth,
            "model_answer": parsed_answer,
            "is_correct": is_correct
        })

    accuracy = correct / total if total > 0 else 0
    print(f"{model_name} Accuracy: {accuracy:.2%}")
    return results, accuracy


# In[4]:


print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")


# ## Use baseline model, test its performance on MedQA dataset, to obtain baseline performance

# In[6]:


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "BioMistral/BioMistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1,
    temperature=0.2,
    device_map="auto",
    do_sample=False
)

def biomistral_fn(prompt):
    output = pipe(prompt)
    return output[0]["generated_text"][len(prompt):].strip()

test_ds = list(dataset["test"])[:200]

biomistral_results, biomistral_accuracy = evaluate_model(biomistral_fn, test_ds, model_name=model_id)

# print(f"BioMistral Results: {biomistral_results[:5]}; Accuracy: {biomistral_accuracy:.2%}")


# In[ ]:




