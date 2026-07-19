"""Small reaction-center graph encoder (pure PyTorch message passing)."""

from __future__ import annotations

from typing import Any

import numpy as np

from catalyst_atlas.featurize.graphs import EDGE_DIM, NODE_DIM
from catalyst_atlas.models.device import require_torch, tensor_to_numpy


def build_mpnn(
    node_dim: int = NODE_DIM,
    edge_dim: int = EDGE_DIM,
    hidden_dim: int = 64,
    embed_dim: int = 64,
    n_layers: int = 3,
    eng_dim: int = 0,
):
    """Construct an MPNN module (requires torch).

    When ``eng_dim > 0``, readout is ``normalize(out([mean_pool; eng]))``.
    """
    torch = require_torch()
    nn = torch.nn
    F = torch.nn.functional

    class MessageLayer(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.msg = nn.Linear(hidden * 3, hidden)
            self.upd = nn.GRUCell(hidden, hidden)

        def forward(self, h, edge_index, e):
            if edge_index.numel() == 0 or e.numel() == 0:
                return h
            src, dst = edge_index[0], edge_index[1]
            msg_in = torch.cat([h[src], h[dst], e], dim=-1)
            msgs = F.relu(self.msg(msg_in))
            agg = torch.zeros_like(h)
            agg = agg.index_add(0, dst, msgs)
            return self.upd(agg, h)

    class MPNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.eng_dim = eng_dim
            self.node_in = nn.Linear(node_dim, hidden_dim)
            self.edge_in = nn.Linear(edge_dim, hidden_dim)
            self.layers = nn.ModuleList([MessageLayer(hidden_dim) for _ in range(n_layers)])
            out_in = hidden_dim + eng_dim
            self.out = nn.Sequential(
                nn.Linear(out_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )

        def forward(self, x, edge_index, edge_attr, eng=None):
            h = F.relu(self.node_in(x))
            if edge_attr.numel() == 0:
                e = x.new_zeros((0, h.size(-1)))
            else:
                e = F.relu(self.edge_in(edge_attr))
            for layer in self.layers:
                h = layer(h, edge_index, e)
            pooled = h.mean(dim=0)
            if self.eng_dim > 0:
                if eng is None:
                    eng = pooled.new_zeros(self.eng_dim)
                else:
                    eng = eng.to(dtype=pooled.dtype, device=pooled.device).view(-1)
                    if eng.numel() != self.eng_dim:
                        raise ValueError(
                            f"eng dim {eng.numel()} != expected eng_dim {self.eng_dim}"
                        )
                pooled = torch.cat([pooled, eng], dim=-1)
            z = self.out(pooled)
            return F.normalize(z, p=2, dim=-1)

    return MPNN()


class ReactionCenterEncoder:
    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_dim: int = 64,
        embed_dim: int = 64,
        n_layers: int = 3,
        eng_dim: int = 0,
    ):
        self.torch = require_torch()
        self.eng_dim = eng_dim
        self.model = build_mpnn(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            n_layers=n_layers,
            eng_dim=eng_dim,
        )
        self.embed_dim = embed_dim

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)

    def encode_graph(self, graph: dict[str, Any], device=None, eng: np.ndarray | None = None):
        torch = self.torch
        x = torch.as_tensor(graph["x"], dtype=torch.float32, device=device)
        ei = torch.as_tensor(graph["edge_index"], dtype=torch.long, device=device)
        ea = torch.as_tensor(graph["edge_attr"], dtype=torch.float32, device=device)
        eng_t = None
        if self.eng_dim > 0:
            if eng is not None:
                eng_t = torch.as_tensor(eng, dtype=torch.float32, device=device)
            elif graph.get("side") is not None and len(np.asarray(graph["side"])) == self.eng_dim:
                eng_t = torch.as_tensor(graph["side"], dtype=torch.float32, device=device)
            else:
                eng_t = torch.zeros(self.eng_dim, dtype=torch.float32, device=device)
        return self.model(x, ei, ea, eng=eng_t)

    def encode_graphs(
        self,
        graphs: list[dict[str, Any]],
        device=None,
        eng_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        out = []
        with torch.no_grad():
            for i, g in enumerate(graphs):
                eng = None if eng_matrix is None else eng_matrix[i]
                emb = self.encode_graph(g, device=device, eng=eng)
                out.append(tensor_to_numpy(emb))
        return np.stack(out, axis=0)
