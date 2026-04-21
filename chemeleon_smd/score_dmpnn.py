"""SCORE-DMPNN: Single-layer DMPNN with contractive recurrent depth.

The original DMPNN stacks 6 message passing layers (shared W_h):
    H_t = relu(H_0 + W_h * M_t)

SCORE-DMPNN uses 1 layer applied for 6 steps with Euler skip connections:
    H_new = relu(H_0 + W_h * M_t)
    H_t   = alpha * H_{t-1} + (1 - alpha) * H_new    (alpha = 0.5, fixed)

1 layer, 6 steps — not 6 layers.

The MolAttFP readout replaces Set2Set (lighter, same dim as input).
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_graphs.utils import scatter


class ScoreDMPNN(nn.Module):
    """Single-layer DMPNN with SCORE contractive recurrent depth.

    Uses 1 shared message passing layer (W_h) applied iteratively for
    ``n_steps`` Euler integration steps with a fixed skip connection alpha.

    This replaces the teacher's 6 stacked layers with 1 layer × 6 steps.
    """

    def __init__(
        self,
        d_v: int = 72,
        d_e: int = 14,
        d_h: int = 2048,
        n_steps: int = 6,
        depth: Optional[int] = None,
        skip_alpha: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_v = d_v
        self.d_e = d_e
        self.d_h = d_h
        self.n_steps = int(depth if depth is not None else n_steps)

        # Single message passing layer
        self.W_i = nn.Linear(d_v + d_e, d_h, bias=False)
        self.W_h = nn.Linear(d_h, d_h, bias=False)
        self.W_o = nn.Linear(d_v + d_h, d_h, bias=True)

        self._skip_alpha = float(skip_alpha)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @property
    def skip_alpha(self) -> float:
        return self._skip_alpha

    @property
    def output_dim(self) -> int:
        return self.d_h

    @property
    def depth(self) -> int:
        return self.n_steps

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

        alpha = self._skip_alpha

        # 1 layer applied for n_steps Euler iterations
        for _ in range(self.n_steps - 1):
            agg_to_dst = scatter(H, dst, out_size=num_atoms, aggr="add")
            msg_at_src = agg_to_dst[src]
            M = msg_at_src - H[rev_edge_index]
            H_new = nn.relu(H_0 + self.W_h(M))
            H = alpha * H + (1.0 - alpha) * H_new
            if self.dropout is not None:
                H = self.dropout(H)

        M_final = scatter(H, dst, out_size=num_atoms, aggr="add")
        output = nn.relu(self.W_o(mx.concatenate([V, M_final], axis=1)))
        if self.dropout is not None:
            output = self.dropout(output)

        return output


class MolAttFPReadout(nn.Module):
    """AttentiveFP-style molecular attention readout (optional).

    Lighter than Set2Set: output_dim == input_dim (not 2x).
    Uses GRU + attention over atoms, T iterative steps.

    NOTE: The default distillation uses mean pooling, not this readout.
    MolAttFPReadout is provided as an experimental alternative whose
    attention weights would need fine-tuning for downstream tasks.
    """

    def __init__(self, hidden_dim: int, num_steps: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.num_steps = num_steps
        self.mol_align = nn.Linear(2 * hidden_dim, 1)
        self.mol_attend = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        scale = 1.0 / (hidden_dim**0.5)
        self.gru_Wx = mx.random.uniform(
            low=-scale, high=scale, shape=(3 * hidden_dim, hidden_dim)
        )
        self.gru_Wh = mx.random.uniform(
            low=-scale, high=scale, shape=(3 * hidden_dim, hidden_dim)
        )
        self.gru_b_ih = mx.zeros((3 * hidden_dim,))
        self.gru_b_hh = mx.zeros((3 * hidden_dim,))

    def _gru_step(self, x: mx.array, h: mx.array) -> mx.array:
        H = self.hidden_dim
        x_proj = x @ self.gru_Wx.T + self.gru_b_ih
        h_proj = h @ self.gru_Wh.T + self.gru_b_hh
        r = mx.sigmoid(x_proj[:, :H] + h_proj[:, :H])
        z = mx.sigmoid(x_proj[:, H : 2 * H] + h_proj[:, H : 2 * H])
        n = mx.tanh(x_proj[:, 2 * H :] + r * h_proj[:, 2 * H :])
        return (1 - z) * n + z * h

    def __call__(
        self,
        node_features: mx.array,
        batch: mx.array,
        num_graphs: int,
    ) -> mx.array:
        mol_feature = scatter(
            node_features, batch, out_size=num_graphs, aggr="add"
        )
        mol_feature = nn.leaky_relu(mol_feature)

        for _ in range(self.num_steps):
            mol_expand = mol_feature[batch]
            align_inp = mx.concatenate([mol_expand, node_features], axis=-1)
            align_score = nn.leaky_relu(self.mol_align(align_inp)).reshape(-1)

            max_per_graph = scatter(
                align_score, batch, out_size=num_graphs, aggr="max"
            )
            attn_w = mx.exp(align_score - max_per_graph[batch])
            norm = scatter(attn_w, batch, out_size=num_graphs, aggr="add")
            attn_w = attn_w / (norm[batch] + 1e-8)

            node_t = self.mol_attend(node_features)
            if self.dropout is not None:
                node_t = self.dropout(node_t)
            mol_context = scatter(
                attn_w[:, None] * node_t,
                batch,
                out_size=num_graphs,
                aggr="add",
            )
            mol_context = nn.elu(mol_context)
            mol_feature = nn.leaky_relu(self._gru_step(mol_feature, mol_context))

        return mol_feature


def load_teacher_dmpnn(
    weights_path: Optional[str] = None,
    config_path: Optional[str] = None,
):
    """Load the pretrained CheMeleon teacher from bundled MLX weights."""
    from chemeleon_smd import convert_weights

    return convert_weights.load_teacher(weights_path=weights_path, config_path=config_path)


def init_student_from_teacher(teacher, skip_alpha: float = 0.0) -> ScoreDMPNN:
    """Initialize a ScoreDMPNN from the teacher's exact weights."""
    student = ScoreDMPNN(
        d_v=teacher.d_v,
        d_e=teacher.d_e,
        d_h=teacher.d_h,
        depth=teacher.depth,
        skip_alpha=skip_alpha,
    )
    student.W_i.weight = teacher.W_i.weight
    student.W_h.weight = teacher.W_h.weight
    student.W_o.weight = teacher.W_o.weight
    student.W_o.bias = teacher.W_o.bias
    return student
