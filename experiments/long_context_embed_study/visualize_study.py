"""Publication-grade Visualizations for Long-Context Legal Retrieval Study.

Generates:
  1. figure1_retrieval_performance.png: P@1, P@5, MRR across candidate models & regimes.
  2. figure2_vram_and_throughput.png: VRAM consumption & throughput scaling on RTX A2000.
  3. figure3_case_score_separation.png: Score distribution across Singularity, Multi-Match, and Zero-Match.
  4. figure4_chunking_granularity.png: Chunk token distribution and context retention.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Set modern aesthetic typography and style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

COLORS = {
    "ModernBERT-base": "#2563EB",       # Royal Blue
    "nomic-embed-text-v1.5": "#059669",  # Emerald Green
    "bge-m3": "#D97706",                 # Amber
    "jina-embeddings-v2-base-en": "#7C3AED", # Violet
    "UltraBERT-Embed-150M": "#DC2626",   # Crimson Red
}


def plot_retrieval_performance(
    benchmark_metrics: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Figure 1: Comparative retrieval performance (P@1, P@5, MRR) across models & regimes."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    regimes = ["micro", "meso", "macro"]
    regime_titles = ["Regime A: Micro (~400t)", "Regime B: Meso (~2,000t)", "Regime C: Macro (Whole-Act)"]

    for ax, reg, reg_title in zip(axes, regimes, regime_titles):
        reg_data = [m for m in benchmark_metrics if m["regime"] == reg]
        models = [m["model_key"] for m in reg_data]

        x = np.arange(len(models))
        width = 0.25

        p1_scores = [m["case1_singularity"]["p_at_1"] for m in reg_data]
        p5_scores = [m["case2_multi_match"]["p_at_5"] for m in reg_data]
        mrr_scores = [m["overall_mrr"] for m in reg_data]

        ax.bar(x - width, p1_scores, width, label="Case 1 P@1", color="#3B82F6", alpha=0.9)
        ax.bar(x, p5_scores, width, label="Case 2 P@5", color="#10B981", alpha=0.9)
        ax.bar(x + width, mrr_scores, width, label="Overall MRR", color="#8B5CF6", alpha=0.9)

        ax.set_title(reg_title, fontweight="bold", pad=10)
        ax.set_xticks(x)
        short_names = [m.split("/")[-1].replace("-base-en", "").replace("-base", "") for m in models]
        ax.set_xticklabels(short_names, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.1)
        ax.set_ylabel("Metric Score (0.0 – 1.0)" if ax == axes[0] else "")
        ax.axhline(1.0, color="#94A3B8", linestyle="--", linewidth=0.8)

    axes[0].legend(loc="lower left", frameon=True)
    plt.suptitle("Retrieval Accuracy Across Chunking Regimes & Candidate Models (9-Query Benchmark)", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_hardware_scaling(
    hardware_data: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Figure 2: Peak VRAM (GB) and Inference Throughput (docs/sec) vs Batch Size on RTX A2000."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    unique_models = list(dict.fromkeys(d["model_key"] for d in hardware_data))

    for m in unique_models:
        m_data = sorted([d for d in hardware_data if d["model_key"] == m], key=lambda x: x["batch_size"])
        batches = [d["batch_size"] for d in m_data]
        vram = [float(d["peak_vram_gb"]) for d in m_data]
        throughput = [float(d["throughput_docs_per_sec"]) for d in m_data]
        color = COLORS.get(m, "#4B5563")

        ax1.plot(batches, vram, marker="o", linewidth=2.0, label=m, color=color)
        ax2.plot(batches, throughput, marker="s", linewidth=2.0, label=m, color=color)

    # 8GB VRAM hardware ceiling line
    ax1.axhline(8.0, color="#EF4444", linestyle=":", linewidth=1.5, label="RTX A2000 Ceiling (8.0 GB)")
    ax1.set_title("Peak GPU VRAM Allocation vs. Batch Size", fontweight="bold")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Peak VRAM (GB)")
    ax1.set_xticks([4, 8, 16, 32])
    ax1.set_ylim(0, 8.5)
    ax1.legend(loc="upper left", frameon=True)

    ax2.set_title("Inference Throughput vs. Batch Size", fontweight="bold")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Throughput (documents / sec)")
    ax2.set_xticks([4, 8, 16, 32])
    ax2.legend(loc="upper left", frameon=True)

    plt.suptitle("Hardware Scaling on NVIDIA RTX A2000 8GB Laptop GPU (CUDA 13.2)", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_case_score_separation(
    benchmark_metrics: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Figure 3: Peak Score Separation across Case 1, Case 2, and Case 3 vs. 0.40 Null Cutoff."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Focus on Macro regime (where long-context models excel)
    macro_data = [m for m in benchmark_metrics if m["regime"] == "macro"]
    models = [m["model_key"] for m in macro_data]

    x = np.arange(len(models))
    width = 0.25

    c1_scores = [m["case1_singularity"]["mean_peak_score"] for m in macro_data]
    c2_scores = [m["case2_multi_match"]["mean_peak_score"] for m in macro_data]
    c3_scores = [m["case3_zero_match"]["mean_peak_score"] for m in macro_data]

    b1 = ax.bar(x - width, c1_scores, width, label="Case 1 (Singularity)", color="#3B82F6", alpha=0.9)
    b2 = ax.bar(x, c2_scores, width, label="Case 2 (Multi-Match)", color="#10B981", alpha=0.9)
    b3 = ax.bar(x + width, c3_scores, width, label="Case 3 (Zero-Match)", color="#EF4444", alpha=0.85)

    # 0.40 Null-rejection threshold line
    ax.axhline(0.40, color="#B91C1C", linestyle="--", linewidth=1.5, label="Calibrated Null Cutoff (0.40)")

    ax.set_title("Peak Similarity Score Distribution Across Search Archetypes (Macro Regime)", fontweight="bold")
    ax.set_xticks(x)
    short_names = [m.split("/")[-1].replace("-base-en", "").replace("-base", "") for m in models]
    ax.set_xticklabels(short_names, rotation=15)
    ax.set_ylabel("Inner Product / Cosine Similarity")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=True)

    # Annotate margin
    for idx, (c1, c3) in enumerate(zip(c1_scores, c3_scores)):
        margin = c1 - c3
        ax.text(idx, c1 + 0.02, f"Δ={margin:.2f}", ha="center", fontsize=9, fontweight="bold", color="#1E293B")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_all_study_figures(
    benchmark_metrics: List[Dict[str, Any]],
    hardware_data: List[Dict[str, Any]],
    figures_dir: Path,
) -> None:
    """Generates and writes all study figures to disk."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_retrieval_performance(benchmark_metrics, figures_dir / "figure1_retrieval_performance.png")
    plot_hardware_scaling(hardware_data, figures_dir / "figure2_vram_and_throughput.png")
    plot_case_score_separation(benchmark_metrics, figures_dir / "figure3_case_score_separation.png")
    print(f"✅ Generated all 3 publication figures in {figures_dir}")
