"""Explicit reaction-center graphs — catalytic machines, not fold fingerprints.

Nodes: catalytic residues, first-shell residues, metals, cofactors.
Edges: pairwise catalytic distances, metal coordination, ligand contacts.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from catalyst_atlas.data.cofactors import COFACTOR_VOCAB
from catalyst_atlas.featurize.features import AA20, AROMATIC, CHARGED, HYDROPHOBIC, POLAR
from catalyst_atlas.paths import PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)

ROLE_VOCAB = ["catalytic", "first_shell", "metal", "cofactor"]
EDGE_TYPE_VOCAB = ["distance", "coordination", "ligand_contact"]
COORD_RADIUS = 3.5
LIGAND_CONTACT_RADIUS = 6.0
MAX_FIRST_SHELL = 12  # keep graphs small and chemistry-focused

NODE_DIM = (
    len(ROLE_VOCAB)  # role
    + len(AA20)  # residue identity
    + 8  # chem proxies
    + len(COFACTOR_VOCAB)  # ligand/metal tag
    + 3  # n_coord, min_dist, dist_to_core (normalized)
)
EDGE_DIM = len(EDGE_TYPE_VOCAB) + 1  # type + distance/10


def _aa_onehot(aa: str) -> np.ndarray:
    vec = np.zeros(len(AA20), dtype=np.float32)
    if aa in AA20:
        vec[AA20.index(aa)] = 1.0
    return vec


def _chem_proxy_aa(aa: str) -> np.ndarray:
    return np.array(
        [
            float(aa in CHARGED),
            float(aa in POLAR),
            float(aa in HYDROPHOBIC),
            float(aa in AROMATIC),
            float(aa == "C"),
            float(aa == "H"),
            float(aa in "DE"),
            float(aa in "KR"),
        ],
        dtype=np.float32,
    )


def _role_onehot(role: str) -> np.ndarray:
    vec = np.zeros(len(ROLE_VOCAB), dtype=np.float32)
    if role in ROLE_VOCAB:
        vec[ROLE_VOCAB.index(role)] = 1.0
    return vec


def _cofactor_onehot(name: str) -> np.ndarray:
    vec = np.zeros(len(COFACTOR_VOCAB), dtype=np.float32)
    tag = name if name in COFACTOR_VOCAB else "none"
    vec[COFACTOR_VOCAB.index(tag)] = 1.0
    return vec


def _edge_type_onehot(kind: str) -> np.ndarray:
    vec = np.zeros(len(EDGE_TYPE_VOCAB), dtype=np.float32)
    if kind in EDGE_TYPE_VOCAB:
        vec[EDGE_TYPE_VOCAB.index(kind)] = 1.0
    return vec


def _residue_node(role: str, aa: str, dist_to_core: float = 0.0) -> np.ndarray:
    return np.concatenate(
        [
            _role_onehot(role),
            _aa_onehot(aa),
            _chem_proxy_aa(aa),
            _cofactor_onehot("none"),
            np.array([0.0, 0.0, float(dist_to_core) / 10.0], dtype=np.float32),
        ]
    ).astype(np.float32)


def _ligand_node(kind: str, name: str, lig: dict[str, Any]) -> np.ndarray:
    role = "metal" if kind == "metal" else "cofactor"
    coord = lig.get("coordination") or {}
    n_coord = float(coord.get("n_coord") or 0) / 10.0
    min_d = float(coord["min_distance"]) / 10.0 if coord.get("min_distance") is not None else 0.0
    dist_core = float(lig.get("dist_to_core") or 0.0) / 10.0
    return np.concatenate(
        [
            _role_onehot(role),
            np.zeros(len(AA20), dtype=np.float32),
            np.zeros(8, dtype=np.float32),
            _cofactor_onehot(name),
            np.array([n_coord, min_d, dist_core], dtype=np.float32),
        ]
    ).astype(np.float32)


def build_reaction_center_graph(microenvironment_json: str | dict[str, Any]) -> dict[str, Any]:
    """Build a fixed-schema graph dict from a microenvironment payload."""
    if isinstance(microenvironment_json, str):
        micro = json.loads(microenvironment_json) if microenvironment_json else {}
    else:
        micro = microenvironment_json or {}

    catalytic = list(micro.get("catalytic") or [])
    first_shell = list(micro.get("first_shell") or [])[:MAX_FIRST_SHELL]
    ligands = list(micro.get("ligands") or [])

    nodes: list[np.ndarray] = []
    xyzs: list[np.ndarray] = []
    node_meta: list[dict[str, Any]] = []

    for r in catalytic:
        nodes.append(_residue_node("catalytic", str(r.get("aa", "X"))))
        xyzs.append(np.asarray(r["xyz"], dtype=float))
        node_meta.append({"kind": "catalytic", "aa": r.get("aa"), "resnum": r.get("resnum")})

    shell_offset = len(nodes)
    for r in first_shell:
        nodes.append(
            _residue_node(
                "first_shell",
                str(r.get("aa", "X")),
                dist_to_core=float(r.get("dist_to_core") or 0.0),
            )
        )
        xyzs.append(np.asarray(r["xyz"], dtype=float))
        node_meta.append({"kind": "first_shell", "aa": r.get("aa"), "resnum": r.get("resnum")})

    lig_offset = len(nodes)
    for lig in ligands:
        kind = str(lig.get("kind") or "cofactor")
        name = str(lig.get("name") or "none")
        nodes.append(_ligand_node(kind, name, lig))
        xyzs.append(np.asarray(lig["xyz"], dtype=float))
        node_meta.append({"kind": kind, "name": name})

    if not nodes:
        # Degenerate empty graph — single zero node so encoders stay well-defined.
        x = np.zeros((1, NODE_DIM), dtype=np.float32)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, EDGE_DIM), dtype=np.float32)
        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "n_nodes": 1,
            "n_edges": 0,
            "node_meta": [{"kind": "empty"}],
        }

    edges_src: list[int] = []
    edges_dst: list[int] = []
    edge_attrs: list[np.ndarray] = []

    def _add_edge(i: int, j: int, kind: str, dist: float) -> None:
        feat = np.concatenate(
            [_edge_type_onehot(kind), np.array([float(dist) / 10.0], dtype=np.float32)]
        )
        # undirected
        for a, b in ((i, j), (j, i)):
            edges_src.append(a)
            edges_dst.append(b)
            edge_attrs.append(feat)

    n_cat = len(catalytic)
    for i in range(n_cat):
        for j in range(i + 1, n_cat):
            d = float(np.linalg.norm(xyzs[i] - xyzs[j]))
            _add_edge(i, j, "distance", d)

    # Catalytic ↔ first-shell contacts within ligand-contact radius
    for si, _r in enumerate(first_shell):
        j = shell_offset + si
        for i in range(n_cat):
            d = float(np.linalg.norm(xyzs[i] - xyzs[j]))
            if d <= LIGAND_CONTACT_RADIUS:
                _add_edge(i, j, "ligand_contact", d)

    # Ligand / metal edges
    for li, lig in enumerate(ligands):
        j = lig_offset + li
        kind = str(lig.get("kind") or "cofactor")
        if kind == "metal":
            coord = lig.get("coordination") or {}
            shell = coord.get("residues") or []
            if shell:
                # Match coordination shell residues to catalytic/first-shell nodes by resnum+aa
                for c in shell:
                    for i, meta in enumerate(node_meta[:lig_offset]):
                        if meta.get("resnum") == c.get("resnum") and meta.get("aa") == c.get("aa"):
                            _add_edge(i, j, "coordination", float(c.get("distance") or COORD_RADIUS))
                            break
            else:
                for i in range(n_cat):
                    d = float(np.linalg.norm(xyzs[i] - xyzs[j]))
                    if d <= COORD_RADIUS:
                        _add_edge(i, j, "coordination", d)
        else:
            for i in range(n_cat):
                d = float(np.linalg.norm(xyzs[i] - xyzs[j]))
                if d <= LIGAND_CONTACT_RADIUS:
                    _add_edge(i, j, "ligand_contact", d)

    x = np.stack(nodes, axis=0)
    if edge_attrs:
        edge_index = np.asarray([edges_src, edges_dst], dtype=np.int64)
        edge_attr = np.stack(edge_attrs, axis=0)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, EDGE_DIM), dtype=np.float32)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "n_nodes": int(x.shape[0]),
        "n_edges": int(edge_attr.shape[0]),
        "node_meta": node_meta,
    }


def graph_to_jsonable(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "x": graph["x"].tolist(),
        "edge_index": graph["edge_index"].tolist(),
        "edge_attr": graph["edge_attr"].tolist(),
        "n_nodes": graph["n_nodes"],
        "n_edges": graph["n_edges"],
        "node_meta": graph["node_meta"],
    }


def graph_from_jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "x": np.asarray(payload["x"], dtype=np.float32),
        "edge_index": np.asarray(payload["edge_index"], dtype=np.int64),
        "edge_attr": np.asarray(payload["edge_attr"], dtype=np.float32),
        "n_nodes": int(payload["n_nodes"]),
        "n_edges": int(payload["n_edges"]),
        "node_meta": payload.get("node_meta") or [],
    }


def build_graphs_table(micro_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build and persist reaction-center graphs for the atlas."""
    ensure_dirs()
    if micro_df is None:
        path = PROCESSED / "microenvironments.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run cat-sites first")
        micro_df = pd.read_parquet(path)

    rows = []
    for _, row in micro_df.iterrows():
        g = build_reaction_center_graph(row.get("microenvironment_json") or "{}")
        rows.append(
            {
                "enzyme_id": row["enzyme_id"],
                "n_nodes": g["n_nodes"],
                "n_edges": g["n_edges"],
                "graph_json": json.dumps(graph_to_jsonable(g)),
                "chemistry_family": row.get("chemistry_family"),
                "mechanistic_pattern": row.get("mechanistic_pattern"),
                "fold_cluster": row.get("fold_cluster"),
                "seq_cluster": row.get("seq_cluster"),
                "ec_number": row.get("ec_number"),
                "catalytic_aas": row.get("catalytic_aas"),
                "sequence": row.get("sequence"),
            }
        )
    out = pd.DataFrame(rows)
    out_path = PROCESSED / "reaction_center_graphs.parquet"
    out.to_parquet(out_path, index=False)
    logger.info(
        "Built %d reaction-center graphs → %s (node_dim=%d edge_dim=%d)",
        len(out),
        out_path,
        NODE_DIM,
        EDGE_DIM,
    )
    return out


def run_graphs() -> dict[str, Any]:
    out = build_graphs_table()
    return {
        "n_enzymes": int(len(out)),
        "mean_nodes": float(out["n_nodes"].mean()),
        "mean_edges": float(out["n_edges"].mean()),
        "node_dim": NODE_DIM,
        "edge_dim": EDGE_DIM,
    }


def load_graphs() -> list[dict[str, Any]]:
    path = PROCESSED / "reaction_center_graphs.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run cat-graphs first")
    df = pd.read_parquet(path)
    graphs = []
    for _, row in df.iterrows():
        g = graph_from_jsonable(json.loads(row["graph_json"]))
        g["enzyme_id"] = row["enzyme_id"]
        g["chemistry_family"] = row.get("chemistry_family")
        g["mechanistic_pattern"] = row.get("mechanistic_pattern")
        g["fold_cluster"] = row.get("fold_cluster")
        g["seq_cluster"] = row.get("seq_cluster")
        g["ec_number"] = row.get("ec_number")
        g["catalytic_aas"] = row.get("catalytic_aas")
        graphs.append(g)
    return graphs
