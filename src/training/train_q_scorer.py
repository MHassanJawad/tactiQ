"""
Train QScorer (Option A): frozen TacticalGNN embeddings -> supervised next-event logits.

Uses the same 11-class labels as graphs from dataset.py (0-9 mapped types, 10 = Other).
Matches inference: q_delta compares predicted logits for actual vs argmax action.
"""

import argparse
import os
import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.models.gnn_encoder import TacticalGNN
from src.models.q_scorer import QScorer


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(graphs: List, val_ratio: float) -> Tuple[List, List]:
    total = len(graphs)
    val_size = max(1, int(total * val_ratio))
    train_size = total - val_size
    if train_size < 1:
        raise ValueError("Dataset too small for train/val split.")
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(graphs, [train_size, val_size], generator=generator)
    return list(train_subset), list(val_subset)


def run_epoch(
    gnn: TacticalGNN,
    scorer: QScorer,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    num_actions: int,
) -> Tuple[float, float]:
    train = optimizer is not None
    scorer.train() if train else scorer.eval()

    total_loss = 0.0
    correct = 0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        targets = batch.y.view(-1).long().clamp(0, num_actions - 1)

        with torch.no_grad():
            z = gnn.encode(batch)

        if train:
            optimizer.zero_grad()

        logits = scorer(z)
        loss = F.cross_entropy(logits, targets)

        if train:
            loss.backward()
            optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        n += bs

    return total_loss / max(1, n), correct / max(1, n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train QScorer on frozen GNN embeddings (next-event CE).")
    p.add_argument("--dataset_path", type=str, default="graphs/la_liga_2015_16_train.pt")
    p.add_argument("--gnn_checkpoint", type=str, default="checkpoints/gnn_best.pt")
    p.add_argument("--output_path", type=str, default="checkpoints/q_scorer_best.pt")
    p.add_argument("--history_path", type=str, default="checkpoints/q_scorer_history.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--hidden_dim", type=int, default=64, help="Must match GNN checkpoint.")
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--gnn_num_classes", type=int, default=11)
    p.add_argument("--num_actions", type=int, default=11, help="QScorer output dim / label space.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    if not os.path.exists(args.gnn_checkpoint):
        raise FileNotFoundError(f"GNN checkpoint not found: {args.gnn_checkpoint}")

    graphs = torch.load(args.dataset_path, map_location="cpu", weights_only=False)
    if not isinstance(graphs, list) or len(graphs) < 10:
        raise ValueError("Expected a list of PyG Data objects with sufficient samples.")

    train_g, val_g = split_dataset(graphs, args.val_ratio)
    train_loader = DataLoader(train_g, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_g, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn = TacticalGNN(
        num_node_features=7,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_classes=args.gnn_num_classes,
    ).to(device)
    gnn.load_state_dict(torch.load(args.gnn_checkpoint, map_location=device, weights_only=False), strict=True)
    gnn.eval()
    for p in gnn.parameters():
        p.requires_grad_(False)

    scorer = QScorer(embedding_dim=args.embed_dim, num_actions=args.num_actions).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.history_path), exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = float("inf")
    stalled = 0

    print(f"Device: {device}")
    print(f"Train graphs: {len(train_g):,} | Val graphs: {len(val_g):,}")
    print(f"Frozen GNN: {args.gnn_checkpoint}")

    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(gnn, scorer, train_loader, device, optimizer, args.num_actions)
        vl, va = run_epoch(gnn, scorer, val_loader, device, None, args.num_actions)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={tl:.4f} acc={ta:.4f} | val_loss={vl:.4f} acc={va:.4f}")

        if vl < best_val:
            best_val = vl
            stalled = 0
            torch.save(scorer.state_dict(), args.output_path)
        else:
            stalled += 1
        if stalled >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    torch.save(history, args.history_path)
    print(f"Saved QScorer checkpoint: {args.output_path}")
    print(f"Saved history: {args.history_path}")


if __name__ == "__main__":
    main()
