import json

import numpy as np
import pandas as pd

from catalyst_atlas.featurize.features import featurize_row
from catalyst_atlas.site.extract import extract_microenvironment


def _toy_row():
    catalytic = [
        {"chain": "A", "resnum": 10, "aa": "S", "role": "catalytic", "xyz": [0.0, 0.0, 0.0]},
        {"chain": "A", "resnum": 35, "aa": "H", "role": "catalytic", "xyz": [3.5, 0.2, 0.1]},
        {"chain": "A", "resnum": 60, "aa": "D", "role": "catalytic", "xyz": [0.3, 3.8, -0.2]},
    ]
    shell = [
        {
            "chain": "A",
            "resnum": 200,
            "aa": "N",
            "role": "first_shell",
            "xyz": [2.0, 2.0, 2.0],
        }
    ]
    return pd.Series(
        {
            "enzyme_id": "TOY001",
            "site_residues_json": json.dumps(catalytic + shell),
            "ligands_json": json.dumps([]),
            "chemistry_class": "hydrolase",
            "catalytic_pattern": "Ser-His-Asp",
            "cofactor_tags": "none",
            "substrate_class": "peptide_ester",
            "ec_number": "3.4.1.1",
            "sequence": "M" * 100,
            "seq_cluster": 1,
            "fold_cluster": 2,
            "family_id": "serine_hydrolase",
            "uniprot_id": "TOY",
            "pdb_id": "TOY",
            "source": "test",
            "is_cryptic_seed": False,
        }
    )


def test_extract_keeps_catalytic_core():
    micro = extract_microenvironment(_toy_row())
    assert micro["n_catalytic"] == 3
    assert micro["catalytic_aas"] == "SHD"
    pairs = json.loads(micro["pairwise_json"])
    assert len(pairs) == 3
    assert all(p["distance"] > 0 for p in pairs)


def test_features_not_composition_only_when_full():
    micro = extract_microenvironment(_toy_row())
    row = pd.Series(micro)
    full = featurize_row(row, composition_only=False)
    comp = featurize_row(row, composition_only=True)
    assert full.shape[0] > comp.shape[0]
    assert np.isfinite(full).all()
