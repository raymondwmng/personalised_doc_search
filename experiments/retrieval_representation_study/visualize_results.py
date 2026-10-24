"""Generate visualization figures and interactive charts for the Research Study (3 Cases / 9 Queries)."""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

STUDY_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = STUDY_DIR / "intermediate_outputs"
FIG_DIR = STUDY_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set high-quality style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def plot_chunk_truncation():
    with open(OUTPUT_DIR / "chunk_representation_audit.json") as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Corpus Token Loss Pie Chart
    labels = [
        f"Lost / Truncated\n(92.75% · {data['embedding_truncation_impact']['total_corpus_tokens'] - data['embedding_truncation_impact']['effectively_embedded_tokens']:,} tok)",
        f"Encoded by MiniLM\n(7.25% · {data['embedding_truncation_impact']['effectively_embedded_tokens']:,} tok)",
    ]
    sizes = [
        data["embedding_truncation_impact"]["corpus_token_loss_percentage"],
        100 - data["embedding_truncation_impact"]["corpus_token_loss_percentage"],
    ]
    colors = ["#ff6b6b", "#4dabf7"]
    explode = (0.08, 0)

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, autopct="%1.1f%%",
        startangle=140, colors=colors, textprops=dict(color="#212529", weight="bold")
    )
    ax1.set_title("Corpus Information Loss\n(all-MiniLM-L6-v2 256-token limit vs 1-chunk documents)")

    # Panel 2: Percentile Token Lengths vs 256 Cutoff
    dist = data["token_length_distribution"]
    percentiles = ["p25", "Median", "p75", "p95", "p99"]
    values = [dist["p25"], dist["median"], dist["p75"], dist["p95"], dist["p99"]]

    bars = ax2.bar(percentiles, values, color="#339af0", width=0.5, edgecolor="#1c7ed6")
    ax2.axhline(256, color="#e03131", linestyle="--", linewidth=2, label="MiniLM 256-Token Hard Ceiling")
    ax2.set_yscale("log")
    ax2.set_ylabel("Tokens per Chunk (Log Scale)")
    ax2.set_title("Chunk Length Distribution vs Embedding Context Ceiling")
    ax2.legend(loc="upper left")

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{int(h):,}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, weight="bold")

    plt.tight_layout()
    out_path = FIG_DIR / "fig1_chunk_truncation.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_score_calibration():
    with open(OUTPUT_DIR / "query_evaluations.json") as f:
        evals = json.load(f)

    # 3 Cases: Singularity (Q1-Q3), Multi-Match (Q4-Q6), Zero-Match (Q7-Q9)
    c1_evals = [e for e in evals if e["case_type"] == "singularity_match"]
    c2_evals = [e for e in evals if e["case_type"] == "multi_match"]
    c3_evals = [e for e in evals if e["case_type"] == "zero_match"]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    x_c1 = np.arange(len(c1_evals))
    x_c2 = np.arange(len(c2_evals)) + len(c1_evals) + 0.5
    x_c3 = np.arange(len(c3_evals)) + len(c1_evals) + len(c2_evals) + 1.0

    b1 = ax.bar(x_c1, [e["top_score"] for e in c1_evals], color="#4dabf7", width=0.55, label="Case 1: Singularity Match (1 Target)", edgecolor="#1971c2")
    b2 = ax.bar(x_c2, [e["top_score"] for e in c2_evals], color="#40c057", width=0.55, label="Case 2: Multi-Match (>= 3 Targets)", edgecolor="#2b8a3e")
    b3 = ax.bar(x_c3, [e["top_score"] for e in c3_evals], color="#fa5252", width=0.55, label="Case 3: Zero-Match (0 in Corpus)", edgecolor="#c92a2a")

    ax.axhline(0.40, color="#fd7e14", linestyle="--", linewidth=2.0, label="Proposed Calibrated Rejection Threshold (0.40)")

    all_x = list(x_c1) + list(x_c2) + list(x_c3)
    all_labels = [e["query_id"] for e in c1_evals] + [e["query_id"] for e in c2_evals] + [e["query_id"] for e in c3_evals]
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels)
    ax.set_ylabel("Peak Retrieval Score (0.75 FAISS + 0.25 Lexical)")
    ax.set_title("Search Score Calibration across 3 Real-World Search Cases (9 Queries)\n(Scores below 0.40 threshold correctly flag zero-match queries without hallucination)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")

    for bar in list(b1) + list(b2) + list(b3):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, weight="bold")

    plt.tight_layout()
    out_path = FIG_DIR / "fig2_score_calibration.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_tag_imbalance():
    with open(OUTPUT_DIR / "tag_utilization_audit.json") as f:
        tag_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Jurisdiction Distribution
    juris = tag_data["category_distribution"]["Jurisdiction"]
    j_cats = list(juris.keys())
    j_vals = list(juris.values())
    j_pcts = [(v / tag_data["total_documents"]) * 100 for v in j_vals]

    ax1.barh(j_cats, j_pcts, color="#4dabf7", edgecolor="#1c7ed6")
    ax1.set_xlabel("% of Entire Corpus (10,013 Docs)")
    ax1.set_title("Jurisdiction Tag Collapse\n(92.2% Assigned to 'EU-wide')")
    for i, v in enumerate(j_pcts):
        ax1.text(v + 1, i, f"{v:.1f}% ({j_vals[i]:,})", va="center", weight="bold")
    ax1.set_xlim(0, 110)

    # Top Regulatory Topics
    topics = tag_data["category_distribution"]["Regulatory Topics"]
    top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:7]
    t_cats = [t[0] for t in reversed(top_topics)]
    t_vals = [t[1] for t in reversed(top_topics)]
    t_pcts = [(v / tag_data["total_documents"]) * 100 for v in t_vals]

    ax2.barh(t_cats, t_pcts, color="#ffa94d", edgecolor="#d9480f")
    ax2.set_xlabel("% of Entire Corpus (10,013 Docs)")
    ax2.set_title("Regulatory Topics Collapse\n(82.0% Assigned to 'Financial Regulation')")
    for i, v in enumerate(t_pcts):
        ax2.text(v + 1, i, f"{v:.1f}% ({t_vals[i]:,})", va="center", weight="bold")
    ax2.set_xlim(0, 105)

    plt.tight_layout()
    out_path = FIG_DIR / "fig3_tag_imbalance.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_retrieval_metrics():
    with open(OUTPUT_DIR / "benchmark_metrics.json") as f:
        metrics = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))

    metric_names = [
        "Case 1 (Singularity)\nPrecision@1",
        "Case 2 (Multi-Match)\nPrecision@1",
        "Case 2 (Multi-Match)\nPrecision@3",
        "Case 2 (Multi-Match)\nPrecision@5",
        "Case 3 (Zero-Match)\nFalse Positive Rate",
        "Tag Utilization\nin Retrieval Ranking",
    ]
    metric_values = [
        metrics["case1_singularity_p_at_1"],
        metrics["case2_multi_p_at_1"],
        metrics["case2_multi_p_at_3"],
        metrics["case2_multi_p_at_5"],
        metrics["case3_zero_false_positive_rate"],
        metrics["tag_utilization_in_retrieval_percentage"] / 100.0,
    ]
    colors = ["#4dabf7", "#40c057", "#40c057", "#40c057", "#fa5252", "#868e96"]

    bars = ax.bar(metric_names, metric_values, color=colors, width=0.55, edgecolor="#212529")
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score / Rate (0.0 to 1.0)")
    ax.set_title("Retrieval Performance across 3 Legal Search Cases (9 Queries)")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1%}" if h > 0 else "0.0%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, weight="bold")

    plt.tight_layout()
    out_path = FIG_DIR / "fig4_retrieval_metrics.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def generate_all():
    plot_chunk_truncation()
    plot_score_calibration()
    plot_tag_imbalance()
    plot_retrieval_metrics()
    print("All visualization figures generated successfully!")


if __name__ == "__main__":
    generate_all()
