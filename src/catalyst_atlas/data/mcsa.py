"""Ingest curated catalytic sites from the M-CSA public API + RCSB structures.

Builds the same atlas schema as the synthetic demo generator so the rest of
the pipeline (sites → embed → eval → search) runs unchanged.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from catalyst_atlas.data.cluster import (
    cath_topology_cluster,
    encode_cluster_ids,
    greedy_sequence_clusters,
)
from catalyst_atlas.data.structures import aa3_to_1, build_site_from_structure, fetch_pdb_text
from catalyst_atlas.paths import RAW, ensure_dirs

logger = logging.getLogger(__name__)

MCSA_ENTRIES_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json"
MCSA_RESIDUES_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/?format=json"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{uid}.fasta"

EC_TO_CHEMISTRY = {
    "1": "oxidoreductase",
    "2": "transferase",
    "3": "hydrolase",
    "4": "lyase",
    "5": "isomerase",
    "6": "ligase",
    "7": "translocase",
}


def _session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "catalyst-atlas/0.1 (research; M-CSA ingest)"})
    return sess


def _cache_dir() -> Path:
    ensure_dirs()
    path = RAW / "mcsa_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_json(url: str, sess: requests.Session, cache_path: Path | None = None) -> Any:
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text())
    resp = sess.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if cache_path is not None:
        cache_path.write_text(json.dumps(data))
    return data


def fetch_all_entries(sess: requests.Session, cache_dir: Path) -> list[dict[str, Any]]:
    """Paginate the M-CSA entries API (~1003 entries)."""
    all_path = cache_dir / "entries_all.json"
    if all_path.exists():
        data = json.loads(all_path.read_text())
        logger.info("Loaded cached M-CSA entries (%d)", len(data))
        return data

    results: list[dict[str, Any]] = []
    url: str | None = MCSA_ENTRIES_URL
    page = 1
    while url:
        logger.info("Fetching M-CSA entries page %d", page)
        payload = fetch_json(url, sess)
        results.extend(payload.get("results") or [])
        url = payload.get("next")
        page += 1
        time.sleep(0.1)
    all_path.write_text(json.dumps(results))
    logger.info("Fetched %d M-CSA entries", len(results))
    return results


def fetch_residues(sess: requests.Session, cache_dir: Path) -> list[dict[str, Any]]:
    path = cache_dir / "residues.json"
    if path.exists():
        data = json.loads(path.read_text())
        logger.info("Loaded cached M-CSA residues (%d)", len(data))
        return data
    logger.info("Fetching M-CSA catalytic residues…")
    data = fetch_json(MCSA_RESIDUES_URL, sess, cache_path=path)
    logger.info("Fetched %d catalytic residue records", len(data))
    return data


def fetch_uniprot_sequence(
    uniprot_id: str, sess: requests.Session, cache_dir: Path
) -> str:
    if not uniprot_id:
        return ""
    path = cache_dir / "sequences" / f"{uniprot_id}.fa"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        text = path.read_text()
        return "".join(
            line.strip() for line in text.splitlines() if line and not line.startswith(">")
        )
    url = UNIPROT_FASTA_URL.format(uid=uniprot_id)
    try:
        resp = sess.get(url, timeout=60)
        if resp.status_code != 200:
            logger.debug("UniProt miss %s (%s)", uniprot_id, resp.status_code)
            return ""
        path.write_text(resp.text)
        return "".join(
            line.strip() for line in resp.text.splitlines() if line and not line.startswith(">")
        )
    except requests.RequestException as exc:
        logger.debug("UniProt error %s: %s", uniprot_id, exc)
        return ""


def chemistry_from_ec(ec: str | None) -> str:
    if not ec:
        return "unknown"
    primary = str(ec).split(".")[0]
    return EC_TO_CHEMISTRY.get(primary, "unknown")


def pattern_from_residues(aas: list[str]) -> str:
    """Compact catalytic pattern from ordered catalytic AAs (e.g. Asp-His-Ser)."""
    if not aas:
        return "unknown"
    # Keep order of annotation; collapse to unique consecutive for readability.
    names = []
    aa3 = {
        "A": "Ala",
        "R": "Arg",
        "N": "Asn",
        "D": "Asp",
        "C": "Cys",
        "Q": "Gln",
        "E": "Glu",
        "G": "Gly",
        "H": "His",
        "I": "Ile",
        "L": "Leu",
        "K": "Lys",
        "M": "Met",
        "F": "Phe",
        "P": "Pro",
        "S": "Ser",
        "T": "Thr",
        "W": "Trp",
        "Y": "Tyr",
        "V": "Val",
    }
    for aa in aas:
        names.append(aa3.get(aa, aa))
    if len(names) > 5:
        # Cap long sites: most frequent catalytic residues.
        top = [a for a, _ in Counter(aas).most_common(4)]
        names = [aa3.get(a, a) for a in top]
    return "-".join(names)


def _reference_chain(residue: dict[str, Any]) -> dict[str, Any] | None:
    chains = residue.get("residue_chains") or []
    for c in chains:
        if c.get("is_reference"):
            return c
    return chains[0] if chains else None


def _group_residues_by_entry(
    residues: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in residues:
        by_id[int(r["mcsa_id"])].append(r)
    return by_id


def build_mcsa_atlas(
    n_enzymes: int | None = None,
    seed: int = 7,
    seq_cluster_threshold: float = 0.15,
    max_pdb_failures: int | None = None,
) -> pd.DataFrame:
    """Download M-CSA + PDB microenvironments and return an atlas DataFrame."""
    cache = _cache_dir()
    pdb_cache = cache / "pdb"
    sess = _session()

    entries = fetch_all_entries(sess, cache)
    residues = fetch_residues(sess, cache)
    by_res = _group_residues_by_entry(residues)
    entry_by_id = {int(e["mcsa_id"]): e for e in entries}

    # Stable order for reproducibility.
    mcsa_ids = sorted(set(by_res) & set(entry_by_id))
    if n_enzymes is not None:
        mcsa_ids = mcsa_ids[: int(n_enzymes)]

    rows: list[dict[str, Any]] = []
    failures = 0
    for mcsa_id in tqdm(mcsa_ids, desc="M-CSA sites+PDB"):
        entry = entry_by_id[mcsa_id]
        res_list = by_res[mcsa_id]
        specs = []
        cath_ids = []
        pdb_id = None
        for r in res_list:
            chain = _reference_chain(r)
            if not chain:
                continue
            pdb_id = pdb_id or chain.get("pdb_id")
            aa = aa3_to_1(str(chain.get("code") or ""))
            specs.append(
                {
                    "chain": chain.get("chain_name") or chain.get("assembly_chain_name") or "A",
                    "resnum": int(chain.get("auth_resid") or chain.get("resid") or 0),
                    "aa": aa,
                }
            )
            if chain.get("domain_cath_id"):
                cath_ids.append(chain["domain_cath_id"])

        if not specs or not pdb_id:
            failures += 1
            continue

        pdb_text = fetch_pdb_text(str(pdb_id), pdb_cache)
        if not pdb_text:
            failures += 1
            if max_pdb_failures is not None and failures >= max_pdb_failures:
                logger.error("Too many PDB failures; aborting early")
                break
            continue

        catalytic, neighbors = build_site_from_structure(pdb_text, specs)
        if len(catalytic) < 2:
            failures += 1
            continue

        uniprot = entry.get("reference_uniprot_id") or ""
        sequence = fetch_uniprot_sequence(str(uniprot), sess, cache)
        if len(sequence) < 20:
            # Fallback: CA-trace order from the mapped structure residues.
            from catalyst_atlas.data.structures import parse_ca_atoms

            sequence = "".join(a["aa"] for a in parse_ca_atoms(pdb_text) if a["aa"] != "X")
        ec_list = entry.get("all_ecs") or []
        ec = (entry.get("reaction") or {}).get("ec") or (ec_list[0] if ec_list else "")
        chem = chemistry_from_ec(ec)
        cat_aas = [r["aa"] for r in catalytic]
        cath = Counter(cath_ids).most_common(1)[0][0] if cath_ids else "unknown"

        rows.append(
            {
                "enzyme_id": f"MCSA{mcsa_id:05d}",
                "uniprot_id": uniprot or f"UNK{mcsa_id}",
                "pdb_id": str(pdb_id).lower(),
                "family_id": f"mcsa_{mcsa_id}",
                "chemistry_class": chem,
                "catalytic_pattern": pattern_from_residues(cat_aas),
                "cofactor_tags": "none",
                "substrate_class": "mcsa_curated",
                "ec_number": str(ec) if ec else "",
                "sequence": sequence,
                "seq_cluster": -1,  # filled below
                "fold_cluster": -1,  # filled below
                "site_residues_json": json.dumps(catalytic + neighbors),
                "ligands_json": json.dumps([]),
                "source": "mcsa",
                "is_cryptic_seed": False,
                "mcsa_id": mcsa_id,
                "enzyme_name": entry.get("enzyme_name") or "",
                "cath_topology": cath_topology_cluster(cath),
            }
        )
        time.sleep(0.02)  # be polite to RCSB / UniProt

    if not rows:
        raise RuntimeError(
            "M-CSA ingest produced zero enzymes — check network / API availability"
        )

    df = pd.DataFrame(rows)
    # Real fold clusters from CATH topology.
    df["fold_cluster"] = encode_cluster_ids(df["cath_topology"].tolist())
    # Real sequence neighborhoods from k-mer Jaccard clustering.
    seq_labels = greedy_sequence_clusters(
        df["sequence"].fillna("").tolist(),
        threshold=seq_cluster_threshold,
        metric="jaccard",
    )
    df["seq_cluster"] = seq_labels

    # Mark cryptic-ish seeds: chemistry differs from majority of their seq cluster.
    chem_by_cluster: dict[int, Counter] = defaultdict(Counter)
    for _, row in df.iterrows():
        chem_by_cluster[int(row["seq_cluster"])][row["chemistry_class"]] += 1
    cryptic = []
    for _, row in df.iterrows():
        maj = chem_by_cluster[int(row["seq_cluster"])].most_common(1)[0][0]
        cryptic.append(row["chemistry_class"] != maj)
    df["is_cryptic_seed"] = cryptic

    logger.info(
        "M-CSA atlas: %d enzymes (%d PDB/map failures skipped); "
        "%d chemistry classes; %d seq clusters; %d fold clusters",
        len(df),
        failures,
        df["chemistry_class"].nunique(),
        df["seq_cluster"].nunique(),
        df["fold_cluster"].nunique(),
    )
    # Drop helper column not in demo schema (keep via parquet is fine; eval uses subset).
    return df
