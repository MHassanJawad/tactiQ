# RT-FSAS Interface Contracts

> These contracts act as the strict boundary between Track A (ML/Data) and Track B (LLM/App). Both members must develop against these exact signatures.

## 🅰️ Built by Member A, Called by Member B

### 1. Game State Encoder (`src/models/gnn_encoder.py`)
```python
import torch
from torch_geometric.data import Data

class TacticalGNN(torch.nn.Module):
    def encode(self, graph: Data) -> torch.Tensor:
        """
        Takes a single match snapshot graph and returns its 128-dim tactical embedding.
        Args:
            graph: PyG Data object containing x, edge_index, edge_attr (23 nodes × 9 features).
        Returns:
            Tensor of shape (128,) representing the latent tactical state.
        """
        pass
```

### 2. Situation Retriever (`src/retrieval/retriever.py`)
```python
import torch
from typing import List, Dict

class TacticalRetriever:
    def retrieve(self, query_embedding: torch.Tensor, k: int = 5) -> List[Dict]:
        """
        Queries the FAISS index for the top-k most structurally similar historical snapshots.
        Args:
            query_embedding: Tensor of shape (128,)
            k: number of similar matches to retrieve
        Returns:
            List of metadata dicts for the top-k matches.
            Example dict: {"match_id": 123, "minute": 45, "formation": "4-3-3", "next_event": "Pass"}
        """
        pass
```

### 3. Decision Quality Scorer (`src/models/q_scorer.py`)
```python
import torch

class QScorer(torch.nn.Module):
    def compute_delta(self, embedding: torch.Tensor, actual_action: int) -> float:
        """
        Computes the Q-delta (suboptimality) for the action taken by the player.
        Args:
            embedding: 128-dim state embedding
            actual_action: integer index of the actual action class (0-14)
        Returns:
            Float representing ΔQ = Q(state, optimal) - Q(state, actual)
            Negative value = suboptimal action.
        """
        pass
```


## 🅱️ Built by Member B, Defines the Final Pipeline

### 4. Gemini Coaching Advisor (`src/llm/gemini_coach.py`)
```python
from typing import List, Dict

class GeminiCoach:
    def advise(self, game_state: Dict, q_delta: float, retrieved_situations: List[Dict]) -> str:
        """
        Generates tactical advice using the RAG retrieved context and Q-delta signal.
        Args:
            game_state: Dict containing current minute, score, team
            q_delta: scalar value output from QScorer
            retrieved_situations: list of dicts output from TacticalRetriever
        Returns:
            String containing the formatted natural language coaching advice.
        """
        pass
```

### 5. Full Pipeline (`src/llm/pipeline.py`)
```python
from typing import Dict, List

class RTFSASPipeline:
    def process(self, live_events: List[Dict], current_minute: int) -> Dict:
        """
        End-to-end processing for the dashboard to call.
        Extracts recent events -> builds graph -> encodes -> retrieves -> scores -> advises.
        Args:
            live_events: recent sequence of match events
            current_minute: the current game minute
        Returns:
            Dict containing:
                "q_delta": float,
                "retrieved": list of retrieved dicts,
                "advice": final string text from Gemini
        """
        pass
```
