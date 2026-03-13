"""Molecule featurizer matching ChemProp's SimpleMoleculeMolGraphFeaturizer.

Produces the same atom (d_v=72) and bond (d_e=14) features as ChemProp v2,
so the pretrained CheMeleon weights work correctly.
"""

from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol, MolFromSmiles
from rdkit.Chem.rdchem import BondStereo, BondType, HybridizationType

ATOM_V2_ATOMIC_NUMS = list(range(1, 37)) + [53]
ATOM_V2_DEGREES = list(range(6))
ATOM_V2_FORMAL_CHARGES = [-1, -2, 1, 2, 0]
ATOM_V2_CHIRAL_TAGS = list(range(4))
ATOM_V2_NUM_HS = list(range(5))
ATOM_V2_HYBRIDIZATIONS = [
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP2D,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
]

ATOM_V2_SUBFEATS = [
    {v: i for i, v in enumerate(ATOM_V2_ATOMIC_NUMS)},
    {v: v for v in ATOM_V2_DEGREES},
    {v: i for i, v in enumerate(ATOM_V2_FORMAL_CHARGES)},
    {v: v for v in ATOM_V2_CHIRAL_TAGS},
    {v: v for v in ATOM_V2_NUM_HS},
    {ht: i for i, ht in enumerate(ATOM_V2_HYBRIDIZATIONS)},
]

ATOM_V2_DIM = sum(1 + len(sf) for sf in ATOM_V2_SUBFEATS) + 2  # +aromatic +mass
BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
BOND_STEREOS = list(range(6))
BOND_DIM = 1 + len(BOND_TYPES) + 2 + (len(BOND_STEREOS) + 1)  # = 14


def featurize_atom(atom: Chem.Atom) -> np.ndarray:
    """72-dim atom feature vector matching ChemProp MultiHotAtomFeaturizer.v2."""
    x = np.zeros(ATOM_V2_DIM, dtype=np.float32)
    feats = [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        int(atom.GetTotalNumHs()),
        atom.GetHybridization(),
    ]
    i = 0
    for feat, choices in zip(feats, ATOM_V2_SUBFEATS):
        j = choices.get(feat, len(choices))
        x[i + j] = 1
        i += len(choices) + 1
    x[i] = int(atom.GetIsAromatic())
    x[i + 1] = 0.01 * atom.GetMass()
    return x


def _one_hot_index(x, xs):
    n = len(xs)
    try:
        return xs.index(x), n + 1
    except (ValueError, AttributeError):
        for idx, val in enumerate(xs):
            if x == val:
                return idx, n + 1
        return n, n + 1


def featurize_bond(bond: Chem.Bond) -> np.ndarray:
    """14-dim bond feature vector matching ChemProp MultiHotBondFeaturizer."""
    x = np.zeros(BOND_DIM, dtype=np.float32)
    if bond is None:
        x[0] = 1
        return x

    i = 1
    bt_bit, size = _one_hot_index(bond.GetBondType(), BOND_TYPES)
    if bt_bit != size:
        x[i + bt_bit] = 1
    i += size - 1

    x[i] = int(bond.GetIsConjugated())
    x[i + 1] = int(bond.IsInRing())
    i += 2

    stereo_bit, _ = _one_hot_index(int(bond.GetStereo()), BOND_STEREOS)
    x[i + stereo_bit] = 1
    return x


class MolGraphData:
    """Featurized molecular graph matching ChemProp's BatchMolGraph format."""

    __slots__ = ("V", "E", "edge_index", "rev_edge_index")

    def __init__(
        self,
        V: np.ndarray,
        E: np.ndarray,
        edge_index: np.ndarray,
        rev_edge_index: np.ndarray,
    ):
        self.V = V
        self.E = E
        self.edge_index = edge_index
        self.rev_edge_index = rev_edge_index


def featurize_mol(mol: Mol) -> Optional[MolGraphData]:
    """Convert an RDKit mol to a MolGraphData matching ChemProp format."""
    if mol is None:
        return None
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    atom_features = np.array(
        [featurize_atom(mol.GetAtomWithIdx(i)) for i in range(n_atoms)],
        dtype=np.float32,
    )

    n_bonds = mol.GetNumBonds()
    if n_bonds == 0:
        return MolGraphData(
            V=atom_features,
            E=np.zeros((0, BOND_DIM), dtype=np.float32),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            rev_edge_index=np.zeros(0, dtype=np.int64),
        )

    src_list, dst_list = [], []
    bond_features = []
    rev_map = {}

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = featurize_bond(bond)
        fwd_idx = len(src_list)
        src_list.append(i)
        dst_list.append(j)
        bond_features.append(bf)
        rev_idx = len(src_list)
        src_list.append(j)
        dst_list.append(i)
        bond_features.append(bf)
        rev_map[fwd_idx] = rev_idx
        rev_map[rev_idx] = fwd_idx

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    E = np.array(bond_features, dtype=np.float32)
    rev_edge_index = np.array(
        [rev_map[i] for i in range(len(src_list))], dtype=np.int64
    )
    return MolGraphData(
        V=atom_features, E=E, edge_index=edge_index, rev_edge_index=rev_edge_index
    )


def featurize_smiles(smiles: str) -> Optional[MolGraphData]:
    """Featurize a SMILES string into a MolGraphData."""
    return featurize_mol(MolFromSmiles(smiles))


def collate_mol_graphs(graphs: List[MolGraphData]) -> Tuple[np.ndarray, ...]:
    """Batch multiple MolGraphData into arrays for the MPNN.

    Returns:
        (V, E, edge_index, rev_edge_index, batch, num_graphs)
    """
    all_V, all_E = [], []
    all_src, all_dst = [], []
    all_rev = []
    batch_assignment = []

    atom_offset = 0
    edge_offset = 0

    for g_idx, g in enumerate(graphs):
        n_atoms = g.V.shape[0]
        n_edges = g.E.shape[0]

        all_V.append(g.V)
        if n_edges > 0:
            all_E.append(g.E)
            all_src.append(g.edge_index[0] + atom_offset)
            all_dst.append(g.edge_index[1] + atom_offset)
            all_rev.append(g.rev_edge_index + edge_offset)
        batch_assignment.extend([g_idx] * n_atoms)

        atom_offset += n_atoms
        edge_offset += n_edges

    V = np.concatenate(all_V, axis=0)
    E = (
        np.concatenate(all_E, axis=0)
        if all_E
        else np.zeros((0, BOND_DIM), dtype=np.float32)
    )
    if all_src:
        src = np.concatenate(all_src)
        dst = np.concatenate(all_dst)
        edge_index = np.stack([src, dst], axis=0)
        rev_edge_index = np.concatenate(all_rev)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        rev_edge_index = np.zeros(0, dtype=np.int64)
    batch = np.array(batch_assignment, dtype=np.int64)

    return V, E, edge_index, rev_edge_index, batch, len(graphs)
