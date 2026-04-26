import os
import argparse
from typing import Dict, List, Optional

import torch

from src.data.graph_builder import GraphBuilder
from src.llm.gemini_coach import GeminiCoach
from src.models.gnn_encoder import TacticalGNN
from src.models.q_scorer import QScorer
from src.retrieval.retriever import TacticalRetriever


ACTION_TO_ID = {
    "Pass": 0,
    "Ball Receipt*": 1,
    "Carry": 2,
    "Pressure": 3,
    "Duel": 4,
    "Clearance": 5,
    "Foul Committed": 6,
    "Interception": 7,
    "Dribble": 8,
    "Shot": 9,
}


class RTFSASPipeline:
    def __init__(
        self,
        gnn_checkpoint_path: str = "checkpoints/gnn_best.pt",
        index_dir: str = "index",
        retriever_k: int = 5,
        gnn_hidden_dim: int = 64,
        gnn_embed_dim: int = 128,
        gnn_num_classes: int = 11,
        device: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ) -> None:
        self.retriever_k = retriever_k
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.graph_builder = GraphBuilder(connect_radius=25.0)
        self.gnn = TacticalGNN(
            num_node_features=7,
            hidden_dim=gnn_hidden_dim,
            embed_dim=gnn_embed_dim,
            num_classes=gnn_num_classes,
        ).to(self.device)
        self.q_scorer = QScorer()
        self.coach = GeminiCoach(api_key=gemini_api_key)

        if not os.path.exists(gnn_checkpoint_path):
            raise FileNotFoundError(
                f"GNN checkpoint missing at {gnn_checkpoint_path}. Train first or pass correct path."
            )
        state = torch.load(gnn_checkpoint_path, map_location=self.device, weights_only=False)
        self.gnn.load_state_dict(state, strict=True)
        self.gnn.eval()

        self.retriever = TacticalRetriever(index_dir=index_dir)

    @torch.no_grad()
    def _encode_current_state(self, event: Dict) -> torch.Tensor:
        graph = self.graph_builder.build_from_event(event, next_action_label=None)
        graph = graph.to(self.device)
        emb = self.gnn.encode(graph)
        # gnn.encode returns shape (1, 128) for single graph with constructed batch
        if emb.dim() == 2:
            emb = emb.squeeze(0)
        return emb.detach().cpu()

    def _event_to_action_id(self, event: Dict) -> int:
        name = event.get("type", {}).get("name", "Pass")
        return ACTION_TO_ID.get(name, 0)

    def process(self, live_events: List[Dict], current_minute: int) -> Dict:
        """
        End-to-end processing: events -> graph -> embedding -> retrieval -> q-delta -> advice.
        """
        if not live_events:
            raise ValueError("live_events is empty; need at least one current event.")

        current_event = dict(live_events[-1])
        if "minute" not in current_event:
            current_event["minute"] = current_minute

        embedding = self._encode_current_state(current_event)
        retrieved = self.retriever.retrieve(embedding, k=self.retriever_k)
        actual_action = self._event_to_action_id(current_event)
        q_delta = self.q_scorer.compute_delta(embedding, actual_action=actual_action)

        game_state = {
            "minute": current_minute,
            "score": current_event.get("score", "unknown"),
            "team": current_event.get("team", {}).get("name", "unknown"),
            "event_type": current_event.get("type", {}).get("name", "Unknown"),
        }
        advice = self.coach.advise(game_state=game_state, q_delta=q_delta, retrieved_situations=retrieved)

        return {
            "q_delta": q_delta,
            "retrieved": retrieved,
            "advice": advice,
        }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RTFSAS pipeline demo.")
    p.add_argument("--checkpoint_path", type=str, default="checkpoints/gnn_best.pt")
    p.add_argument("--index_dir", type=str, default="index")
    p.add_argument("--retriever_k", type=int, default=5)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=11)
    p.add_argument("--gemini_api_key", type=str, default=None)
    p.add_argument("--minute", type=int, default=67)
    p.add_argument("--event_type", type=str, default="Pass")
    p.add_argument("--team_name", type=str, default="Team A")
    p.add_argument("--x", type=float, default=82.0)
    p.add_argument("--y", type=float, default=25.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    pipeline = RTFSASPipeline(
        gnn_checkpoint_path=args.checkpoint_path,
        index_dir=args.index_dir,
        retriever_k=args.retriever_k,
        gnn_hidden_dim=args.hidden_dim,
        gnn_embed_dim=args.embed_dim,
        gnn_num_classes=args.num_classes,
        gemini_api_key=args.gemini_api_key,
    )
    live_events = [
        {
            "location": [args.x, args.y],
            "minute": args.minute,
            "type": {"name": args.event_type},
            "team": {"name": args.team_name},
            "possession_team": {"id": 1},
            "score": "unknown",
            "match_id": 0,
        }
    ]
    out = pipeline.process(live_events, current_minute=args.minute)
    print(f"q_delta: {out['q_delta']:.4f}")
    print(f"retrieved_count: {len(out['retrieved'])}")
    print("advice:")
    print(out["advice"])


if __name__ == "__main__":
    main()

