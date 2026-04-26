import argparse
import json
import os
import random
from statistics import mean
from typing import Any, Dict, List

import torch

from src.llm.gemini_coach import GeminiCoach
from src.models.gnn_encoder import TacticalGNN
from src.models.q_scorer import QScorer
from src.retrieval.retriever import TacticalRetriever


def _load_graphs(graphs_path: str) -> List[Any]:
    if os.path.isfile(graphs_path):
        graphs = torch.load(graphs_path, map_location="cpu", weights_only=False)
        if not isinstance(graphs, list):
            raise ValueError("graphs_path file must contain a list of graphs.")
        return graphs

    if os.path.isdir(graphs_path):
        shard_files = sorted(
            [
                os.path.join(graphs_path, name)
                for name in os.listdir(graphs_path)
                if name.startswith("la_liga_2015_16_full_part_") and name.endswith(".pt")
            ]
        )
        if not shard_files:
            raise ValueError(f"No shard files found in {graphs_path}")
        graphs: List[Any] = []
        for shard_path in shard_files:
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            if not isinstance(shard, list):
                raise ValueError(f"Shard is not a list: {shard_path}")
            graphs.extend(shard)
        return graphs

    raise FileNotFoundError(f"graphs_path does not exist: {graphs_path}")


def _specificity_score(text: str) -> int:
    keywords = [
        "minute",
        "space",
        "press",
        "pass",
        "shot",
        "wing",
        "midfield",
        "defensive",
        "turnover",
        "similar",
        "historical",
        "risk",
    ]
    low = text.lower()
    return sum(1 for kw in keywords if kw in low)


def _safe_event_type_from_label(label: int) -> str:
    labels = {
        0: "Pass",
        1: "Ball Receipt*",
        2: "Carry",
        3: "Pressure",
        4: "Duel",
        5: "Clearance",
        6: "Foul Committed",
        7: "Interception",
        8: "Dribble",
        9: "Shot",
    }
    return labels.get(int(label), "Pass")


@torch.no_grad()
def run_ablation(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    graphs = _load_graphs(args.graphs_path)
    if not graphs:
        raise ValueError("No graphs loaded.")

    sample_size = min(args.sample_size, len(graphs))
    sampled = rng.sample(graphs, sample_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn = TacticalGNN(
        num_node_features=7,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
    ).to(device)
    state = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    gnn.load_state_dict(state, strict=True)
    gnn.eval()

    retriever = TacticalRetriever(index_dir=args.index_dir)
    scorer = QScorer()
    coach = GeminiCoach(api_key=args.gemini_api_key)

    retrieval_top1_hits = 0
    retrieval_topk_hits = 0
    top1_sims: List[float] = []
    advice_len_words_full: List[int] = []
    advice_len_words_dumb: List[int] = []
    advice_spec_full: List[int] = []
    advice_spec_dumb: List[int] = []

    for g in sampled:
        graph = g.to(device)
        emb = gnn.encode(graph)
        if emb.dim() == 2:
            emb = emb.squeeze(0)
        emb = emb.detach().cpu()

        retrieved = retriever.retrieve(emb, k=args.k)
        actual_label = int(g.y.view(-1)[0].item()) if getattr(g, "y", None) is not None else 0
        actual_action = min(actual_label, scorer.num_actions - 1)
        q_delta = scorer.compute_delta(emb, actual_action=actual_action)

        top1 = retrieved[0] if retrieved else {}
        top1_label = int(top1.get("next_event_label", -999))
        if top1_label == actual_label:
            retrieval_top1_hits += 1
        if any(int(row.get("next_event_label", -999)) == actual_label for row in retrieved):
            retrieval_topk_hits += 1
        if "similarity" in top1:
            top1_sims.append(float(top1["similarity"]))

        game_state = {
            "minute": int(getattr(g, "minute", 0)),
            "score": "unknown",
            "team": "Team A",
            "event_type": str(getattr(g, "event_type", _safe_event_type_from_label(actual_label))),
        }

        advice_full = coach.advise(game_state, q_delta=q_delta, retrieved_situations=retrieved)
        advice_dumb = coach.advise(game_state, q_delta=0.0, retrieved_situations=[])

        advice_len_words_full.append(len(advice_full.split()))
        advice_len_words_dumb.append(len(advice_dumb.split()))
        advice_spec_full.append(_specificity_score(advice_full))
        advice_spec_dumb.append(_specificity_score(advice_dumb))

    out = {
        "num_samples": sample_size,
        "retrieval": {
            "top1_next_event_match_rate": retrieval_top1_hits / sample_size,
            "topk_next_event_contains_rate": retrieval_topk_hits / sample_size,
            "avg_top1_similarity": mean(top1_sims) if top1_sims else 0.0,
        },
        "advice_quality": {
            "full_avg_words": mean(advice_len_words_full),
            "dumb_avg_words": mean(advice_len_words_dumb),
            "full_avg_specificity_score": mean(advice_spec_full),
            "dumb_avg_specificity_score": mean(advice_spec_dumb),
        },
    }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RT-FSAS ablation metrics.")
    p.add_argument("--graphs_path", type=str, default="graphs/la_liga_2015_16_full_shards")
    p.add_argument("--checkpoint_path", type=str, default="checkpoints/gnn_best.pt")
    p.add_argument("--index_dir", type=str, default="index")
    p.add_argument("--output_path", type=str, default="report/ablation_metrics.json")
    p.add_argument("--sample_size", type=int, default=50)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=11)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gemini_api_key", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_ablation(args)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {args.output_path}")


if __name__ == "__main__":
    main()

