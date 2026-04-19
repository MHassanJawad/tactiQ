"""
FAISS-backed similarity search over tactical graph embeddings.

Loads artifacts from src/retrieval/build_index.py:
  index/tactical.index
  index/tactical_meta.pt
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import faiss


class TacticalRetriever:
    def __init__(
        self,
        index_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        index_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            index_path: Full path to tactical.index (default: index/tactical.index under cwd).
            meta_path: Full path to tactical_meta.pt
            index_dir: If set, uses index_dir/tactical.index and index_dir/tactical_meta.pt
        """
        root = os.getcwd()
        if index_dir is not None:
            idx = os.path.join(index_dir, "tactical.index")
            meta = os.path.join(index_dir, "tactical_meta.pt")
        else:
            idx = index_path or os.path.join(root, "index", "tactical.index")
            meta = meta_path or os.path.join(root, "index", "tactical_meta.pt")

        if not os.path.isfile(idx):
            raise FileNotFoundError(f"FAISS index not found: {idx}. Run src/retrieval/build_index.py first.")
        if not os.path.isfile(meta):
            raise FileNotFoundError(f"Metadata not found: {meta}. Run src/retrieval/build_index.py first.")

        self._index = faiss.read_index(idx)
        raw = torch.load(meta, map_location="cpu", weights_only=False)
        if not isinstance(raw, list):
            raise ValueError("tactical_meta.pt must contain a list of dicts.")
        self._metadata: List[Dict[str, Any]] = raw

    def retrieve(self, query_embedding: torch.Tensor, k: int = 5) -> List[Dict[str, Any]]:
        """
        Queries the FAISS index for the top-k most similar historical snapshots.

        Args:
            query_embedding: Tensor of shape (128,) or (1, 128) — same space as build_index (L2-normalized IP).
            k: Number of neighbors.

        Returns:
            List of metadata dicts with added keys: similarity (inner product), rank (1-based).
        """
        if k < 1:
            return []

        q = query_embedding.detach().cpu().numpy().astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self._index.d:
            raise ValueError(
                f"Query dim {q.shape[1]} != index dim {self._index.d}. Expected 128-d tactical embedding."
            )

        faiss.normalize_L2(q)
        k_eff = min(k, self._index.ntotal)
        if k_eff == 0:
            return []

        scores, indices = self._index.search(q, k_eff)
        out: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            row = dict(self._metadata[idx])
            row["similarity"] = float(score)
            row["rank"] = rank
            out.append(row)
        return out
