"""Unit tests for M-CSA structure parsing and sequence clustering (offline)."""

from __future__ import annotations

from catalyst_atlas.data.cluster import (
    cath_topology_cluster,
    greedy_sequence_clusters,
    kmer_jaccard,
)
from catalyst_atlas.data.mcsa import chemistry_from_ec, pattern_from_residues
from catalyst_atlas.data.structures import build_site_from_structure, parse_ca_atoms

_MINI_PDB = """\
HEADER    TEST
ATOM      1  N   ASP A   7      11.000  12.000  13.000  1.00 20.00           N
ATOM      2  CA  ASP A   7      10.000  20.000  30.000  1.00 20.00           C
ATOM      3  CA  HIS A  35      13.500  20.200  30.100  1.00 20.00           C
ATOM      4  CA  SER A  60      10.300  23.800  29.800  1.00 20.00           C
ATOM      5  CA  ASN A 200      12.000  22.000  32.000  1.00 20.00           C
ATOM      6  CA  LEU A 300      40.000  40.000  40.000  1.00 20.00           C
END
"""


def test_parse_ca_atoms():
    atoms = parse_ca_atoms(_MINI_PDB)
    assert len(atoms) == 5
    assert atoms[0]["aa"] == "D"
    assert atoms[0]["resnum"] == 7


def test_build_site_from_structure():
    specs = [
        {"chain": "A", "resnum": 7, "aa": "D"},
        {"chain": "A", "resnum": 35, "aa": "H"},
        {"chain": "A", "resnum": 60, "aa": "S"},
    ]
    catalytic, neighbors = build_site_from_structure(_MINI_PDB, specs)
    assert len(catalytic) == 3
    assert "".join(r["aa"] for r in catalytic) == "DHS"
    assert any(n["aa"] == "N" for n in neighbors)
    assert not any(n["resnum"] == 300 for n in neighbors)  # far LEU excluded


def test_chemistry_from_ec_and_pattern():
    assert chemistry_from_ec("3.4.21.1") == "hydrolase"
    assert chemistry_from_ec("1.1.1.1") == "oxidoreductase"
    assert pattern_from_residues(["S", "H", "D"]) == "Ser-His-Asp"


def test_cath_and_seq_clustering():
    assert cath_topology_cluster("3.40.50.1860") == "3.40.50"
    assert cath_topology_cluster(None) == "unknown"
    assert kmer_jaccard("ACDEFGHIK", "ACDEFGHIK") == 1.0
    labels = greedy_sequence_clusters(
        ["ACDEFGHIKLMNPQRSTVWY" * 3, "ACDEFGHIKLMNPQRSTVWY" * 3, "QQQQQQQQQQ" * 5],
        threshold=0.5,
        metric="containment",
    )
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]
