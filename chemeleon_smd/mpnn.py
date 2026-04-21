"""CheMeleon MPNN (BondMessagePassing) in MLX — Teacher model.

Faithfully reproduces ChemProp's BondMessagePassing with depth=6
message passing iterations and shared W_h:

  h_vw^0 = relu(W_i * [x_v; e_vw])
  m_vw^t = sum_{u in N(v) \\ w} h_uv^(t-1)
  h_vw^t = relu(h_vw^0 + W_h * m_vw^t)        (shared W_h, hard overwrite)
  m_v^T  = sum_{w in N(v)} h_wv^T
  h_v^T  = relu(W_o * [x_v; m_v^T])

Hyperparameters from pretrained CheMeleon:
  d_v=72, d_e=14, d_h=2048, depth=6, activation=relu, bias=False
  Total message passing params: 8,714,240 (8.7M)

Note: ChemProp already shares W_h across all iterations.
The difference with SCORE-DMPNN is in the forward dynamics:
  Teacher:  H_t = relu(H_0 + W_h * M_t)               (hard overwrite)
  SCORE:    H_t = 0.5*H_{t-1} + 0.5*relu(H_0 + W_h*M_t)  (Euler skip)
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_graphs.utils import scatter


class CheMeleonBondMPNN(nn.Module):
    """ChemProp-style Directed Bond Message Passing Neural Network in MLX."""

    def __init__(
        self,
        d_v: int = 72,
        d_e: int = 14,
        d_h: int = 2048,
        depth: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_v = d_v
        self.d_e = d_e
        self.d_h = d_h
        self.depth = depth

        self.W_i = nn.Linear(d_v + d_e, d_h, bias=False)
        self.W_h = nn.Linear(d_h, d_h, bias=False)
        self.W_o = nn.Linear(d_v + d_h, d_h, bias=True)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @property
    def output_dim(self) -> int:
        return self.d_h

    def __call__(
        self,
        V: mx.array,
        E: mx.array,
        edge_index: mx.array,
        rev_edge_index: mx.array,
    ) -> mx.array:
        src = edge_index[0]
        dst = edge_index[1]
        num_atoms = V.shape[0]

        H_0 = self.W_i(mx.concatenate([V[src], E], axis=1))
        H = nn.relu(H_0)

        for _ in range(1, self.depth):
            agg_to_dst = scatter(H, dst, out_size=num_atoms, aggr="add")
            msg_at_src = agg_to_dst[src]
            M = msg_at_src - H[rev_edge_index]
            H = nn.relu(H_0 + self.W_h(M))
            if self.dropout is not None:
                H = self.dropout(H)

        M_final = scatter(H, dst, out_size=num_atoms, aggr="add")
        output = nn.relu(self.W_o(mx.concatenate([V, M_final], axis=1)))
        if self.dropout is not None:
            output = self.dropout(output)

        return output


class Set2SetReadout(nn.Module):
    """Set2Set graph-level readout producing 2 * input_dim representations."""

    def __init__(self, input_dim: int, n_iters: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.n_iters = n_iters
        self.output_dim = 2 * input_dim
        self.lstm_W_i = nn.Linear(2 * input_dim, 4 * input_dim)
        self.lstm_W_h = nn.Linear(input_dim, 4 * input_dim, bias=False)

    def _lstm_step(
        self,
        x: mx.array,
        h: mx.array,
        c: mx.array,
    ) -> tuple[mx.array, mx.array]:
        d_h = self.input_dim
        gates = self.lstm_W_i(x) + self.lstm_W_h(h)
        i = mx.sigmoid(gates[:, :d_h])
        f = mx.sigmoid(gates[:, d_h : 2 * d_h])
        g = mx.tanh(gates[:, 2 * d_h : 3 * d_h])
        o = mx.sigmoid(gates[:, 3 * d_h :])
        c_new = f * c + i * g
        h_new = o * mx.tanh(c_new)
        return h_new, c_new

    def __call__(
        self,
        node_features: mx.array,
        batch: mx.array,
        num_graphs: int,
    ) -> mx.array:
        d_h = self.input_dim
        h = mx.zeros((num_graphs, d_h))
        c = mx.zeros((num_graphs, d_h))
        q_star = mx.zeros((num_graphs, 2 * d_h))

        for _ in range(self.n_iters):
            h, c = self._lstm_step(q_star, h, c)
            query_per_node = h[batch]
            scores = mx.sum(query_per_node * node_features, axis=-1)
            alpha = scatter(scores, batch, out_size=num_graphs, aggr="softmax")
            readout = scatter(
                mx.expand_dims(alpha, -1) * node_features,
                batch,
                out_size=num_graphs,
                aggr="add",
            )
            q_star = mx.concatenate([h, readout], axis=-1)

        return q_star
