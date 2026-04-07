import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class TacticalGNN(torch.nn.Module):
    def __init__(self, num_node_features=7, hidden_dim=64, embed_dim=128, num_classes=11):
        """
        A Graph Attention Network designed to learn football tactics.
        Args:
            num_node_features: 7 (x, y, team_binary, gk, def, mid, fwd)
            hidden_dim: The internal processing size (kept small for CPU speed)
            embed_dim: The final output vector size for the FAISS index (128)
            num_classes: How many event types we predict (Pass, Shot, etc.)
        """
        super(TacticalGNN, self).__init__()
        
        # 1. First Graph Attention Layer
        # Processes 7 raw features (x,y,team,roles) into hidden dimensions
        self.conv1 = GATv2Conv(num_node_features, hidden_dim, heads=2, concat=False)
        
        # 2. Second Graph Attention Layer
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False)
        
        # 3. Final embedding layer before pooling
        self.conv3 = GATv2Conv(hidden_dim, embed_dim, heads=1, concat=False)
        
        # 4. Classification Head (only used for training)
        # We predict the next action type (0-10) to force the network to learn valid football tactics
        self.classifier = torch.nn.Linear(embed_dim, num_classes)

    def encode(self, data):
        """
        Takes raw graph components and squashes them into a single 128-dim vector.
        THIS is the exact method Partner B will call in their final application pipeline.
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, getattr(data, 'batch', None)
        
        # Message passing round 1
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Message passing round 2
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Message passing round 3 (embedding round)
        x = self.conv3(x, edge_index, edge_attr)
        
        # If there's no batch vector (e.g. real-time inference on 1 graph), create a batch of 0s
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # Global Mean Pooling
        # Averages the 23 individual player vectors into 1 master graph vector (shape: [batch_size, 128])
        graph_embed = global_mean_pool(x, batch)
        return graph_embed

    def forward(self, data):
        """
        Full forward pass used exclusively during supervised training.
        """
        # Get the 128-dim tactical fingerprint
        graph_embed = self.encode(data)
        
        # Pass it through the classifier to predict the next event type
        out = self.classifier(graph_embed)
        return out
