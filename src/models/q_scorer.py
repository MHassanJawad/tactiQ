import torch
import torch.nn as nn
import torch.nn.functional as F


class QScorer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        num_actions: int = 11,
        dropout: float = 0.2,
    ) -> None:
        """
        Lightweight MLP scorer for action values Q(state, action).

        Args:
            embedding_dim: Input tactical embedding size from TacticalGNN.
            hidden_dim_1: First hidden layer width.
            hidden_dim_2: Second hidden layer width.
            num_actions: Number of discrete action classes.
            dropout: Dropout probability used between hidden layers.
        """
        super().__init__()
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, num_actions),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Returns per-action Q-values.

        Args:
            embedding: Tensor of shape (128,) or (batch_size, 128)

        Returns:
            Tensor of shape (num_actions,) or (batch_size, num_actions)
        """
        is_single = embedding.dim() == 1
        if is_single:
            embedding = embedding.unsqueeze(0)
        q_values = self.net(embedding)
        return q_values.squeeze(0) if is_single else q_values

    @torch.no_grad()
    def compute_delta(self, embedding: torch.Tensor, actual_action: int) -> float:
        """
        Computes delta for the observed action against the best available action.

        Convention used here:
            q_delta = Q(actual_action) - max_a Q(a)
        This means:
            - 0.0  => observed action is optimal
            - < 0  => observed action is suboptimal

        Args:
            embedding: 128-dim state embedding.
            actual_action: Integer action index (0 to num_actions - 1).

        Returns:
            Float q_delta score.
        """
        if actual_action < 0 or actual_action >= self.num_actions:
            raise ValueError(
                f"actual_action={actual_action} is out of range [0, {self.num_actions - 1}]"
            )

        self.eval()
        q_values = self.forward(embedding)
        if q_values.dim() != 1 or q_values.size(0) != self.num_actions:
            raise ValueError(
                f"Expected Q-values shape ({self.num_actions},), got {tuple(q_values.shape)}"
            )

        q_actual = q_values[actual_action]
        q_best = torch.max(q_values)
        q_delta = q_actual - q_best
        return float(q_delta.item())

