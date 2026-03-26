import torch
from torch_geometric.data import Data
import numpy as np
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore")

class GraphBuilder:
    def __init__(self, connect_radius=25.0):
        """
        Builds PyTorch Geometric graphs from StatsBomb events.
        Without 360 tracking data, we estimate off-ball player positions based on tactical roles.
        
        Args:
            connect_radius: float, distance in spatial units to draw an edge between two players.
        """
        self.connect_radius = connect_radius
        
        # Base normalized grid positions (0-1 pitch) for standard 4-3-3 as a fallback.
        # Format: role_name -> (x_norm, y_norm)
        self.role_baselines = {
            "Goalkeeper": (0.05, 0.5),
            "Right Back": (0.25, 0.85),
            "Right Center Back": (0.2, 0.65),
            "Left Center Back": (0.2, 0.35),
            "Left Back": (0.25, 0.15),
            "Defensive Midfield": (0.4, 0.5),
            "Right Center Midfield": (0.5, 0.7),
            "Left Center Midfield": (0.5, 0.3),
            "Right Wing": (0.75, 0.85),
            "Left Wing": (0.75, 0.15),
            "Center Forward": (0.8, 0.5),
        }

    def _estimate_positions(self, event_x, event_y, possession_team_id):
        """
        Estimates the 22 player positions. 
        Shifts the base formation towards the ball's center of gravity.
        Returns a list of node features:
        [x, y, team_binary, is_gk, is_def, is_mid, is_fwd]
        """
        nodes = []
        
        # We'll mock the 11 home and 11 away players for now.
        # In a full pipeline, we map actual lineup IDs to these nodes.
        
        # Normalize event coordinates (Statsbomb pitch is 120 x 80)
        ball_x_norm = min(max(event_x / 120.0, 0), 1)
        ball_y_norm = min(max(event_y / 80.0, 0), 1)
        
        # Team 0 (Possession Team) - shifted towards the ball
        for role, (bx, by) in self.role_baselines.items():
            # Blend base position with ball gravity
            ex = (bx * 0.7) + (ball_x_norm * 0.3)
            ey = (by * 0.7) + (ball_y_norm * 0.3)
            
            # Simple one-hot role logic
            is_gk = 1.0 if "Goalkeeper" in role else 0.0
            is_def = 1.0 if "Back" in role else 0.0
            is_mid = 1.0 if "Midfield" in role else 0.0
            is_fwd = 1.0 if "Forward" in role or "Wing" in role else 0.0
            
            nodes.append([ex, ey, 0.0, is_gk, is_def, is_mid, is_fwd])

        # Team 1 (Defending Team) - mirrored and slightly deeper
        for role, (bx, by) in self.role_baselines.items():
            ex = 1.0 - ((bx * 0.8) + (ball_x_norm * 0.2)) # Defending, so inverse direction
            ey = 1.0 - ((by * 0.8) + (ball_y_norm * 0.2))
            
            is_gk = 1.0 if "Goalkeeper" in role else 0.0
            is_def = 1.0 if "Back" in role else 0.0
            is_mid = 1.0 if "Midfield" in role else 0.0
            is_fwd = 1.0 if "Forward" in role or "Wing" in role else 0.0
            
            nodes.append([ex, ey, 1.0, is_gk, is_def, is_mid, is_fwd])
            
        return np.array(nodes, dtype=np.float32)

    def build_from_event(self, event_dict, next_action_label=None):
        """
        Converts a single statsbomb JSON event dictionary into a PyG Data object.
        """
        # Default center pitch if location is missing (rare but happens)
        loc = event_dict.get('location', [60.0, 40.0])
        x, y = loc[0], loc[1]
        team_id = event_dict.get('possession_team', {}).get('id', 0)
        
        # 1. Get Node Features
        node_features = self._estimate_positions(x, y, team_id)
        
        # Add the ball itself as the 23rd node (team 0.5, no roles)
        ball_node = np.array([[x/120.0, y/80.0, 0.5, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        node_features = np.vstack([node_features, ball_node])
        
        x_tensor = torch.tensor(node_features, dtype=torch.float)
        
        # 2. Build Edges (Distance-based connectivity)
        # Convert coords back to pitch scale for distance check
        pitch_scale = np.array([120.0, 80.0])
        real_coords = node_features[:, :2] * pitch_scale
        
        edge_indices = []
        edge_attrs = []
        
        num_nodes = len(node_features)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist = np.linalg.norm(real_coords[i] - real_coords[j])
                    if dist <= self.connect_radius:
                        edge_indices.append([i, j])
                        # Normalize edge distance as attribute [0, 1]
                        edge_attrs.append([dist / self.connect_radius])
                        
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            # Fallback if graph is totally disconnected
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
            
        # 3. Target Label (for the GNN supervised training later)
        # Event Type integer class (0 to 14)
        y_tensor = torch.tensor([next_action_label], dtype=torch.long) if next_action_label is not None else None
            
        # Assemble Graph
        data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
        
        # Store metadata for retrieval later
        data.match_id = event_dict.get("match_id", 0)
        data.minute = event_dict.get("minute", 0)
        data.event_type = event_dict.get("type", {}).get("name", "Unknown")
        
        return data

# Quick test if run directly
if __name__ == "__main__":
    builder = GraphBuilder(connect_radius=30.0)
    mock_event = {
        "location": [85.0, 20.0],
        "possession_team": {"id": 217},
        "type": {"name": "Pass"},
        "minute": 14,
        "match_id": 999
    }
    
    graph = builder.build_from_event(mock_event, next_action_label=2)
    print(f"Graph generated successfully!")
    print(f"Nodes geometry: {graph.x.shape}") 
    print(f"Edges geometry: {graph.edge_index.shape}")
    print(f"Mock target label: {graph.y.item()}")
