import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _load_metrics(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_single_panel_series(metrics: Dict) -> Tuple[List[str], List[float]]:
    retrieval = metrics.get("retrieval", {})
    advice = metrics.get("advice_quality", {})

    labels = [
        "Top-1 match rate",
        "Top-k contain rate",
        "Top-1 similarity",
        "Full specificity",
        "Dumb specificity",
        "Full words (scaled)",
        "Dumb words (scaled)",
    ]

    # Keep one chart readable by scaling word counts.
    values = [
        float(retrieval.get("top1_next_event_match_rate", 0.0)),
        float(retrieval.get("topk_next_event_contains_rate", 0.0)),
        float(retrieval.get("avg_top1_similarity", 0.0)),
        float(advice.get("full_avg_specificity_score", 0.0)) / 10.0,
        float(advice.get("dumb_avg_specificity_score", 0.0)) / 10.0,
        float(advice.get("full_avg_words", 0.0)) / 100.0,
        float(advice.get("dumb_avg_words", 0.0)) / 100.0,
    ]
    return labels, values


def _build_two_panel_series(metrics: Dict) -> Tuple[List[str], List[float], List[str], List[float]]:
    retrieval = metrics.get("retrieval", {})
    advice = metrics.get("advice_quality", {})

    retrieval_labels = [
        "Top-1 match",
        "Top-k contain",
        "Top-1 similarity",
    ]
    retrieval_values = [
        float(retrieval.get("top1_next_event_match_rate", 0.0)),
        float(retrieval.get("topk_next_event_contains_rate", 0.0)),
        float(retrieval.get("avg_top1_similarity", 0.0)),
    ]

    advice_labels = [
        "Full words",
        "Dumb words",
        "Full specificity",
        "Dumb specificity",
    ]
    advice_values = [
        float(advice.get("full_avg_words", 0.0)),
        float(advice.get("dumb_avg_words", 0.0)),
        float(advice.get("full_avg_specificity_score", 0.0)),
        float(advice.get("dumb_avg_specificity_score", 0.0)),
    ]
    return retrieval_labels, retrieval_values, advice_labels, advice_values


def _annotate_bars(ax, bars, fmt: str = "{:.2f}", offset: float = 0.01) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def generate_figure(metrics_path: str, output_png: str, output_svg: str = "", two_panel: bool = False) -> None:
    metrics = _load_metrics(metrics_path)

    if two_panel:
        r_labels, r_values, a_labels, a_values = _build_two_panel_series(metrics)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.0))

        bars1 = ax1.bar(r_labels, r_values)
        ax1.set_ylim(0, 1.1)
        ax1.set_ylabel("Rate / Similarity")
        ax1.set_title("Figure 6(a): Retrieval Metrics")
        _annotate_bars(ax1, bars1, fmt="{:.2f}", offset=0.02)

        bars2 = ax2.bar(a_labels, a_values)
        ax2.set_ylabel("Value")
        ax2.set_title("Figure 6(b): Advice Quality Metrics")
        _annotate_bars(ax2, bars2, fmt="{:.2f}", offset=max(0.02, 0.01 * max(a_values or [1.0])))

        for tick in ax1.get_xticklabels():
            tick.set_rotation(0)
        for tick in ax2.get_xticklabels():
            tick.set_rotation(0)

        fig.tight_layout()
    else:
        labels, values = _build_single_panel_series(metrics)
        plt.figure(figsize=(10, 5.2))
        bars = plt.bar(labels, values)
        plt.ylim(0, 1.1)
        plt.ylabel("Normalized value (0-1 scale)")
        plt.title("Figure 6: Ablation Metrics (Full vs Baseline Proxies)")
        plt.xticks(rotation=20, ha="right")
        _annotate_bars(plt.gca(), bars, fmt="{:.2f}", offset=0.02)
        plt.tight_layout()

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    if output_svg:
        os.makedirs(os.path.dirname(output_svg), exist_ok=True)
        plt.savefig(output_svg, bbox_inches="tight")
    plt.close()

    print(f"Saved Figure 6 PNG: {output_png}")
    if output_svg:
        print(f"Saved Figure 6 SVG: {output_svg}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Figure 6 from ablation_metrics.json")
    p.add_argument("--metrics_path", type=str, default="report/ablation_metrics.json")
    p.add_argument("--output_png", type=str, default="report/fig6_ablation_metrics.png")
    p.add_argument("--output_svg", type=str, default="report/fig6_ablation_metrics.svg")
    p.add_argument(
        "--two_panel",
        action="store_true",
        help="Generate IEEE-style two-panel figure (retrieval + advice).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_figure(args.metrics_path, args.output_png, args.output_svg, two_panel=args.two_panel)


if __name__ == "__main__":
    main()

