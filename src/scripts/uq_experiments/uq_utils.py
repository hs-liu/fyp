# uq_utils.py
import numpy as np
import pandas as pd
from collections import Counter


def compute_uq(prompt: str, inference_fn, n_samples, temperature) -> dict:

    from scripts.baselines.baseline_utils import parse_answer

    samples = [
        parse_answer(inference_fn(prompt, do_sample=True, temperature=temperature))
        for _ in range(n_samples)
    ]

    valid = [s for s in samples if s != "UNKNOWN"]

    if not valid:
        return {
            "uq_answer":      "UNKNOWN",
            "uq_consistency": 0.0,
            "uq_entropy":     1.0,
            "uq_samples":     str(samples),
            "n_valid":        0,
        }

    counts   = Counter(valid)
    majority, majority_count = counts.most_common(1)[0]
    consistency = majority_count / len(valid)
    probs   = np.array([c / len(valid) for c in counts.values()])
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

    return {
        "uq_answer":      majority,
        "uq_consistency": round(consistency, 3),
        "uq_entropy":     round(entropy, 3),
        "uq_samples":     str(samples),
        "n_valid":        len(valid),
    }


def analyse_uq(df: pd.DataFrame, answer_col: str = "uq_answer",
               correct_col: str = "uq_correct") -> dict:
    """
    Compute calibration and coverage-accuracy tradeoff from UQ results.
    
    Returns dict with calibration table and tradeoff curve.
    """
    # calibration by consistency bin
    bins   = [0.0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["<0.4", "0.4-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    df = df.copy()
    df["consistency_bin"] = pd.cut(df["uq_consistency"], bins=bins, labels=labels)

    calib = df.groupby("consistency_bin", observed=True).agg(
        n=(correct_col, "count"),
        accuracy=(correct_col, "mean"),
        avg_consistency=("uq_consistency", "mean"),
    ).reset_index()

    # coverage vs accuracy tradeoff
    tradeoff = []
    for threshold in np.arange(0.1, 1.01, 0.05):
        subset = df[df["uq_consistency"] >= threshold]
        if len(subset) < 5:
            break
        tradeoff.append({
            "threshold": round(threshold, 2),
            "coverage":  round(len(subset) / len(df), 3),
            "accuracy":  round(subset[correct_col].mean(), 3),
            "n":         len(subset),
        })

    return {
        "calibration": calib,
        "tradeoff":    pd.DataFrame(tradeoff),
        "overall_acc": df[correct_col].mean(),
        "greedy_acc":  df["greedy_correct"].mean() if "greedy_correct" in df else None,
    }


def plot_uq(df: pd.DataFrame, save_path: str, model_name: str = "Model"):
    """Plot calibration curve and coverage-accuracy tradeoff."""
    import matplotlib.pyplot as plt

    analysis = analyse_uq(df)
    calib    = analysis["calibration"]
    tradeoff = analysis["tradeoff"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"UQ Analysis — {model_name}", fontsize=13)

    # calibration bar chart
    axes[0].bar(calib["consistency_bin"].astype(str), calib["accuracy"], color="steelblue")
    axes[0].axhline(analysis["overall_acc"], color="red", linestyle="--", label=f"Overall {analysis['overall_acc']:.2%}")
    if analysis["greedy_acc"]:
        axes[0].axhline(analysis["greedy_acc"], color="green", linestyle="--", label=f"Greedy {analysis['greedy_acc']:.2%}")
    axes[0].set_xlabel("Consistency score")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Calibration: Consistency vs Accuracy")
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    # coverage vs accuracy
    axes[1].plot(tradeoff["coverage"], tradeoff["accuracy"], marker="o", color="steelblue")
    axes[1].set_xlabel("Coverage (fraction answered)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Coverage vs Accuracy tradeoff")
    axes[1].invert_xaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")


def print_uq_summary(df: pd.DataFrame):
    """Print a summary of UQ results to stdout."""
    analysis = analyse_uq(df)

    print(f"\nUQ Summary:")
    print(f"  Greedy accuracy:      {analysis['greedy_acc']:.2%}" if analysis['greedy_acc'] else "")
    print(f"  UQ majority accuracy: {analysis['overall_acc']:.2%}")
    print(f"\nCalibration table:")
    print(analysis["calibration"].to_string(index=False))
    print(f"\nCoverage vs Accuracy:")
    print(analysis["tradeoff"].to_string(index=False))