import argparse
import os
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.models.gnn_encoder import TacticalGNN


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(graphs: List[torch.Tensor], val_ratio: float) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    total = len(graphs)
    val_size = max(1, int(total * val_ratio))
    train_size = total - val_size
    if train_size < 1:
        raise ValueError("Dataset too small. Need at least 2 graphs for train/val split.")
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(graphs, [train_size, val_size], generator=generator)
    return list(train_subset), list(val_subset)


def run_epoch(
    model: TacticalGNN,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        targets = batch.y.view(-1).long()

        if is_train:
            optimizer.zero_grad()

        logits = model(batch)
        loss = F.cross_entropy(logits, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = correct / max(1, total_samples)
    return avg_loss, avg_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TacticalGNN to predict next football event type.")
    parser.add_argument("--dataset_path", type=str, default="graphs/la_liga_2015_16_train.pt")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/gnn_best.pt")
    parser.add_argument("--history_path", type=str, default="checkpoints/gnn_history.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at '{args.dataset_path}'. Run src/data/dataset.py first."
        )

    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.history_path), exist_ok=True)

    graphs = torch.load(args.dataset_path, map_location="cpu", weights_only=False)
    if not isinstance(graphs, list) or len(graphs) < 2:
        raise ValueError("Loaded dataset must be a list of PyG Data objects with length >= 2.")

    train_graphs, val_graphs = split_dataset(graphs, args.val_ratio)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TacticalGNN(
        num_node_features=7,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print(f"Device: {device}")
    print(f"Training samples: {len(train_graphs):,} | Validation samples: {len(val_graphs):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    torch.save(history, args.history_path)
    print(f"Best checkpoint saved to: {args.checkpoint_path}")
    print(f"Training history saved to: {args.history_path}")


if __name__ == "__main__":
    main()
