"""Enrich an existing atlas with cofactors + chemistry ontology (offline, cached PDBs)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from catalyst_atlas.data.cofactors import cofactors_near_site
from catalyst_atlas.data.generate_demo import save_raw_atlas
from catalyst_atlas.data.labels import annotate_chemistry
from catalyst_atlas.paths import RAW, ensure_dirs

logger = logging.getLogger(__name__)


def enrich_atlas_cofactors_and_ontology(
    atlas_path: Path | None = None,
    pdb_cache: Path | None = None,
) -> pd.DataFrame:
    """Re-scan cached PDBs for cofactors/metals and attach chemistry ontology labels."""
    ensure_dirs()
    path = atlas_path or (RAW / "catalytic_atlas.parquet")
    cache = pdb_cache or (RAW / "mcsa_cache" / "pdb")
    if not path.exists():
        raise FileNotFoundError(f"Missing atlas at {path}")

    df = pd.read_parquet(path)
    n_with_cof = 0
    families = []
    patterns = []
    cof_tags = []
    ligands_col = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Enrich cofactors/ontology"):
        catalytic = []
        try:
            residues = json.loads(row.get("site_residues_json") or "[]")
            catalytic = [r for r in residues if r.get("role") == "catalytic"]
        except json.JSONDecodeError:
            catalytic = []

        tags = str(row.get("cofactor_tags") or "none")
        ligands: list = []
        pdb_id = str(row.get("pdb_id") or "")
        pdb_path = cache / f"{pdb_id.lower()}.pdb" if pdb_id else None
        if pdb_path and pdb_path.exists() and catalytic:
            pdb_text = pdb_path.read_text(errors="replace")
            try:
                all_res = json.loads(row.get("site_residues_json") or "[]")
            except json.JSONDecodeError:
                all_res = catalytic
            ligands, tags = cofactors_near_site(
                pdb_text, catalytic, radius=8.0, site_residues=all_res
            )
            if tags != "none":
                n_with_cof += 1

        cat_aas = [r.get("aa", "") for r in catalytic]
        ann = annotate_chemistry(
            ec_number=row.get("ec_number"),
            chemistry_class=row.get("chemistry_class"),
            catalytic_aas=cat_aas,
            cofactor_tags=tags,
        )
        families.append(ann["chemistry_family"])
        patterns.append(ann["mechanistic_pattern"])
        cof_tags.append(tags)
        ligands_col.append(json.dumps(ligands))

    df["cofactor_tags"] = cof_tags
    df["ligands_json"] = ligands_col
    df["chemistry_family"] = families
    df["mechanistic_pattern"] = patterns
    # Legacy EC-style mirror for older code paths / demos.
    legacy_map = {
        "hydrolysis": "hydrolase",
        "oxidation-reduction": "oxidoreductase",
        "transfer": "transferase",
        "elimination": "lyase",
        "carbon-carbon chemistry": "lyase",
        "isomerization": "isomerase",
        "ligation": "ligase",
    }
    df["chemistry_class"] = [legacy_map.get(f, "unknown") for f in families]

    save_raw_atlas(df)
    logger.info(
        "Enriched atlas: %d enzymes, %d with site cofactors/metals; families=%s",
        len(df),
        n_with_cof,
        sorted(df["chemistry_family"].unique().tolist()),
    )
    return df
