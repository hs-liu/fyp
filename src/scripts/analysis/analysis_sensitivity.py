import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TEMPERATURES  = [0.1, 0.3, 0.5, 0.7, 0.9]
SAMPLE_SIZES  = [5, 10, 20, 50]
N_FIXED       = 10     # fixed N for temperature sensitivity
T_FIXED       = 0.7    # fixed T for sample size sensitivity

COLORS = {
    "BioMistral-7B": "#E07B54",
    "Llama-3.1-8B":  "#5B8DB8",
    "Qwen2.5-7B":    "#6DBF82",
}

MARKERS = {
    "BioMistral-7B": "o",
    "Llama-3.1-8B":  "s",
    "Qwen2.5-7B":    "^",
}

# ── File maps ─────────────────────────────────────────────────────────────────
#   Adjust these paths/filenames to match your actual file locations.

TEMP_FILES = {
    "BioMistral-7B": {t: f"./results/UQ/results_biomistral_medhireuqrag_{t}_{N_FIXED}.csv" for t in TEMPERATURES},
    "Llama-3.1-8B":  {t: f"./results/UQ/results_llama_medhireuqrag_{t}_{N_FIXED}.csv"      for t in TEMPERATURES},
    "Qwen2.5-7B":    {t: f"./results/UQ/results_qwen_medhireuqrag_{t}_{N_FIXED}.csv"        for t in TEMPERATURES},
}

NSIZE_FILES = {
    "BioMistral-7B": {n: f"./results/UQ/results_biomistral_medhireuqrag_{T_FIXED}_{n}.csv" for n in SAMPLE_SIZES},
    "Llama-3.1-8B":  {n: f"./results/UQ/results_llama_medhireuqrag_{T_FIXED}_{n}.csv"      for n in SAMPLE_SIZES},
    "Qwen2.5-7B":    {n: f"./results/UQ/results_qwen_medhireuqrag_{T_FIXED}_{n}.csv"        for n in SAMPLE_SIZES},
}

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_metrics(filepath):
    """Returns (accuracy %, mean consistency) from a single CSV file."""
    df = pd.read_csv(filepath)
    accuracy    = df["uq_correct"].mean() * 100
    consistency = df["uq_consistency"].mean()
    return accuracy, consistency


def collect_results(file_map, sweep_values):
    """
    Iterates over models and sweep values (temperatures or sample sizes),
    loads each CSV, and returns a nested dict of lists.

    Returns:
        { model_name: { "accuracy": [...], "consistency": [...] } }
    """
    results = {model: {"accuracy": [], "consistency": []} for model in file_map}
    for model, val_files in file_map.items():
        for val in sweep_values:
            acc, cons = load_metrics(val_files[val])
            results[model]["accuracy"].append(acc)
            results[model]["consistency"].append(cons)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING FUNCTION (reused for both analyses)
# ══════════════════════════════════════════════════════════════════════════════

def plot_sensitivity(
    results,
    x_values,
    x_label,
    suptitle,
    save_path,
):
    """
    Generic two-panel sensitivity plot.

    Left panel  → Majority Vote Accuracy (%)
    Right panel → Average Self-Consistency Score
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)

    for model, metrics in results.items():
        c = COLORS[model]
        m = MARKERS[model]

        axes[0].plot(x_values, metrics["accuracy"],
                     marker=m, color=c, linewidth=2, markersize=7, label=model)
        axes[1].plot(x_values, metrics["consistency"],
                     marker=m, color=c, linewidth=2, markersize=7, label=model)

    # ── Left: Accuracy
    axes[0].set_title("Majority Vote Accuracy", fontsize=12)
    axes[0].set_xlabel(x_label, fontsize=11)
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].set_xticks(x_values)
    axes[0].set_ylim(0, 100)
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # ── Right: Consistency
    axes[1].set_title("Average Self-Consistency Score", fontsize=12)
    axes[1].set_xlabel(x_label, fontsize=11)
    axes[1].set_ylabel("Mean Consistency", fontsize=11)
    axes[1].set_xticks(x_values)
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ── 1. Temperature Sensitivity (N fixed = 10) ─────────────────────────────
    print("Running temperature sensitivity analysis...")
    temp_results = collect_results(TEMP_FILES, TEMPERATURES)

    plot_sensitivity(
        results   = temp_results,
        x_values  = TEMPERATURES,
        x_label   = "Temperature",
        suptitle  = f"Temperature Sensitivity Analysis  (N = {N_FIXED})",
        save_path = "./graphs/analysis/uq_analysis/temperature_sensitivity.png",
    )

    # ── 2. Sample Size Sensitivity (T fixed = 0.7) ────────────────────────────
    print("Running sample size sensitivity analysis...")
    nsize_results = collect_results(NSIZE_FILES, SAMPLE_SIZES)

    plot_sensitivity(
        results   = nsize_results,
        x_values  = SAMPLE_SIZES,
        x_label   = "Sample Size (N)",
        suptitle  = f"Sample Size Sensitivity Analysis  (T = {T_FIXED})",
        save_path = "./graphs/analysis/uq_analysis/sample_size_sensitivity.png",
    )

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()
