"""
Encode all cached graphs with the trained TacticalGNN and build a FAISS index for similarity search.

Outputs (default):
  index/tactical.index   — FAISS IndexFlatIP on L2-normalized 128-d embeddings (cosine similarity)
  index/tactical_meta.pt — list[dict] aligned with vector row ids (same order as graphs)
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Project root on path when run as script
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import faiss  # noqa: E402

from src.models.gnn_encoder import TacticalGNN  # noqa: E402


def _label_to_next_event_name(label: int) -> str:
    names = {
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
        10: "Other",
    }
    return names.get(int(label), "Other")


def _graph_metadata(g: Any, row_id: int) -> Dict[str, Any]:
    y = g.y
    if y is None:
        label = -1
    else:
        label = int(y.view(-1)[0].item())
    return {
        "vector_id": row_id,
        "match_id": int(getattr(g, "match_id", 0)),
        "minute": int(getattr(g, "minute", 0)),
        "event_type": str(getattr(g, "event_type", "Unknown")),
        "next_event_label": label,
        "next_event": _label_to_next_event_name(label) if label >= 0 else "Unknown",
        "formation": "unknown",
    }


@torch.no_grad()
def encode_all(
    model: TacticalGNN,
    graphs: List[Any],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    chunks: List[torch.Tensor] = []
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        emb = model.encode(batch)
        chunks.append(emb.cpu())
    if not chunks:
        return np.zeros((0, 128), dtype=np.float32)
    out = torch.cat(chunks, dim=0).numpy().astype(np.float32)
    return np.ascontiguousarray(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS index from trained TacticalGNN embeddings.")
    p.add_argument(
        "--graphs_path",
        type=str,
        default="graphs/la_liga_2015_16_full.pt",
        help="List of PyG Data objects (full index set).",
    )
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/gnn_best.pt",
        help="Trained model weights.",
    )
    p.add_argument(
        "--index_dir",
        type=str,
        default="index",
        help="Directory for tactical.index and tactical_meta.pt",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=11)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.graphs_path):
        raise FileNotFoundError(
            f"Graphs not found: {args.graphs_path}. Run src/data/dataset.py to build .pt files."
        )
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint_path}. Train with python -m src.training.train_gnn first."
        )

    graphs = torch.load(args.graphs_path, map_location="cpu", weights_only=False)
    if not isinstance(graphs, list) or len(graphs) == 0:
        raise ValueError("graphs_path must be a non-empty list of PyG Data objects.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TacticalGNN(
        num_node_features=7,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
    ).to(device)
    state = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)

    print(f"Encoding {len(graphs):,} graphs on {device}...")
    embeddings = encode_all(model, graphs, device, args.batch_size)
    d = embeddings.shape[1]

    metadata = [_graph_metadata(g, i) for i, g in enumerate(graphs)]

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    os.makedirs(args.index_dir, exist_ok=True)
    index_path = os.path.join(args.index_dir, "tactical.index")
    meta_path = os.path.join(args.index_dir, "tactical_meta.pt")

    faiss.write_index(index, index_path)
    torch.save(metadata, meta_path)

    print(f"FAISS index saved: {index_path} (vectors: {index.ntotal}, dim: {d})")
    print(f"Metadata saved: {meta_path} (entries: {len(metadata)})")


if __name__ == "__main__":
    main()
