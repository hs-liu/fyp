import re

import pandas as pd 

def format_question(sample):
    """Format a MedQA sample into a plain-text prompt."""
    options = "\n".join(
    [f"{opt['key']}. {opt['value']}" for opt in sample["options"]]
    )
    return (
        "You are a medical expert. Answer each question with only the letter of the correct answer (A, B, C, D, or E). Do not explain.\n\n"
        
        "Question: A 45-year-old man presents with chest pain. "
        "Which enzyme is most specific for myocardial infarction?\n"
        "A. AST\nB. LDH\nC. Troponin I\nD. CK-MB\n"
        "Answer: C\n\n"
        
        "Question: Which of the following is the first-line treatment for hypertension?\n"
        "A. Beta blockers\nB. ACE inhibitors\nC. Calcium channel blockers\nD. Diuretics\n"
        "Answer: D\n\n"
        
        f"Question: {sample['question']}\n"
        f"{options}\n"
        f"Answer:"          # model should complete with just the letter
    )

def format_options_dict(sample):
    """Return options as {A: text, B: text, ...} for MedRAG."""
    return {opt["key"]: opt["value"] for opt in sample["options"]}

def parse_answer(ans):
    """Extract first A-E letter from raw model output."""
    if not ans:
        return "UNKNOWN"
    # Take only first character of output
    first = ans.strip().upper()[0]
    if first in ["A", "B", "C", "D", "E"]:
        return first
    # Fallback: search for letter
    match = re.search(r'\b([A-E])\b', ans.strip().upper())
    return match.group(1) if match else "UNKNOWN"

def evaluate_model(model_fn, questions, model_name="model", save_path=None, summary_path=None):
    """
    Evaluate a model function over a list of MedQA samples.

    model_fn : callable(sample) -> raw string answer
    questions : list of dataset samples
    model_name : label for logging
    save_path : if provided, saves CSV of results here
    summary_path : if provided, saves summary of results here
    """
    total = len(questions)
    correct = 0
    results = []

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    for i, sample in enumerate(questions):
        try:
            raw_answer = model_fn(sample)
        except Exception as e:
            print(f" [ERROR] sample {i}: {e}")
            raw_answer = ""

        parsed_answer = parse_answer(raw_answer)
        ground_truth = sample["answer_idx"]
        is_correct = parsed_answer == ground_truth
        print("ground truth is", ground_truth, "parsed answer", parsed_answer, "is correct?", is_correct)
        if is_correct:
            correct += 1

        results.append({
            "id" : i,
            "question" : sample["question"],
            "ground_truth" : ground_truth,
            "raw_answer" : raw_answer,
            "model_answer" : parsed_answer,
            "is_correct" : is_correct,
        })

        if (i + 1) % 20 == 0:
            print(f" [{i+1:>3}/{total}] running accuracy: {correct/(i+1):.2%}")

    accuracy = correct / total if total > 0 else 0
    print(f"\n ✓ {model_name} Final Accuracy: {accuracy:.2%} ({correct}/{total})")

    if save_path:
        pd.DataFrame(results).to_csv(save_path, index=False)
        print(f" Results saved → {save_path}")
    
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(f"{model_name} Accuracy: {accuracy:.2%} ({correct}/{total})\n")
        print(f" Accuracy saved → {summary_path}")

    return results, accuracy

def get_retry_delay(error_str, default=65):
    """Parse retry delay from 429 error message."""
    match = re.search(r'retry in (\d+\.?\d*)s', str(error_str))
    return float(match.group(1)) + 5 if match else default