import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax  # sparse row-wise softmax
from torch_geometric.typing import Adj, OptTensor
import sys

class SparseGraphAttentionLayer(nn.Module):
    """
    GAT layer that accepts a sparse COO edge_index (2, E) and
    optional edge weights instead of an (N, N) dense adjacency.
    Works for a single graph or for a PyG batch (with `batch`)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.6,
                 negative_slope: float = 0.2,
                 add_self_loops: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops

        # learnable linear layers
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(size=(out_channels, 1)))
        self.att_dst = nn.Parameter(torch.empty(size=(out_channels, 1)))
        nn.init.xavier_uniform_(self.lin.weight, gain=1.414)
        nn.init.xavier_uniform_(self.att_src, gain=1.414)
        nn.init.xavier_uniform_(self.att_dst, gain=1.414)

    def forward(
        self,
        x: torch.Tensor,             # [N, Fin]
        edge_index: torch.Tensor,    # [2, E]  (COO list of edges)
        edge_weight: OptTensor = None,
        size: tuple = None
    ):
        # ---- 1. Linear projection
        Wh = self.lin(x)             # [N, Fout]

        # ---- 2. Prepare <aᵀ [Wh_i || Wh_j]> for each edge
        # (edge_index[0] are sources, edge_index[1] are destinations)
        Wh_src, Wh_dst = Wh[edge_index[0]], Wh[edge_index[1]]
        e_src = (Wh_src * self.att_src.T).sum(dim=-1)   # [E]
        e_dst = (Wh_dst * self.att_dst.T).sum(dim=-1)   # [E]
        e = F.leaky_relu(e_src + e_dst, self.negative_slope)

        # ---- 3. Sparse softmax normalised per destination node
        #      (works for batched graphs, too)
        alpha = softmax(e, edge_index[1])               # [E]
        alpha = self.dropout(alpha)

        # ---- 4. Message aggregation  (attention-weighted sum)
        if edge_weight is not None:
            alpha = alpha * edge_weight                 # optional weighting

        out = Wh_src * alpha.unsqueeze(-1)              # [E, Fout]
        out = scatter_add(out, edge_index[1], dim=0, dim_size=x.size(0))

        return out, alpha   # node embeddings, raw attention per edge
