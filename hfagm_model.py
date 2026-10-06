import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Set path to import layers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.layers.sparse_gat_layer import SparseGraphAttentionLayer

# === Attribute Attention Module ===
class AttributeAttention(nn.Module):
    def __init__(self, input_dim):
        super(AttributeAttention, self).__init__()
        self.attn_fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_scores = torch.softmax(self.attn_fc(x), dim=1)  # [B, T, 1]
        attended = attn_scores * x                           # [B, T, D]
        return attended.sum(dim=1)                           # [B, D]

# === HFAGM Model ===
class HFAGM(nn.Module):
    def __init__(self,
                 attr_dim: int,
                 graph_dim: int,
                 hidden_dim: int,
                 num_classes: int):
        super(HFAGM, self).__init__()

        self.attribute_attn = AttributeAttention(attr_dim)
        self.attr_fc = nn.Linear(attr_dim, hidden_dim)  # Project attribute to hidden dim

        self.graph_attn = SparseGraphAttentionLayer(graph_dim, hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)  # Combine both pathways

    def forward(self, x_attr, x_graph, adj_matrix):
        # === Attribute pathway ===
        x_attr_attn = self.attribute_attn(x_attr)           # [B, attr_dim]
        attr_repr = self.attr_fc(x_attr_attn)               # [B, hidden_dim]

        # === Graph pathway ===
        graph_repr, attn_weights = self.graph_attn(x_graph, adj_matrix)  # [B, hidden_dim]

        # === Combine both and classify ===
        combined = torch.cat([attr_repr, graph_repr], dim=1)             # [B, 2*hidden_dim]
        logits = self.classifier(combined)                               # [B, num_classes]

        return logits, attn_weights
