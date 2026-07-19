"""Expand the catalytic atlas beyond curated M-CSA.

Adds UniProt-annotated catalytic / active sites with experimental PDB
structures when available, and AlphaFold models as a stratified secondary
track (``structure_source=alphafold``).

Also attaches coarse EC evaluation labels (``ec_class``, ``ec3``).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import pandas as pd
import requests

from catalyst_atlas.data.cluster import encode_cluster_ids, greedy_sequence_clusters
from catalyst_atlas.data.labels import annotate_chemistry
from catalyst_atlas.data.structures import build_site_from_structure, fetch_pdb_text
from catalyst_atlas.paths import RAW, ensure_dirs

logger = logging.getLogger(__name__)

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
AFDB_PDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb"


def _session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "catalyst-atlas/0.4 (research; UniProt expand)"})
    return sess


def _cache_dir():
    ensure_dirs()
    path = RAW / "uniprot_expand_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ec_labels(ec_number: str | None) -> dict[str, str]:
    """Coarse EC class (1–7) and EC to three levels."""
    s = str(ec_number or "").strip()
    parts = [p for p in s.split(".") if p]
    ec_class = parts[0] if parts else "unknown"
    ec3 = ".".join(parts[:3]) if parts else "unknown"
    return {"ec_class": ec_class, "ec3": ec3}


def attach_ec_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    labs = [ec_labels(e) for e in out.get("ec_number", pd.Series([""] * len(out)))]
    out["ec_class"] = [x["ec_class"] for x in labs]
    out["ec3"] = [x["ec3"] for x in labs]
    if "structure_source" not in out.columns:
        # M-CSA / demo rows are experimental PDB by default.
        out["structure_source"] = out.get("source", pd.Series(["unknown"] * len(out))).map(
            lambda s: "experimental" if str(s) in {"mcsa", "demo_high_confidence"} else "experimental"
        )
    return out


def _paginate_uniprot_search(
    query: str,
    n_enzymes: int,
    sess: requests.Session,
    cache_name: str,
) -> list[dict[str, Any]]:
    """Paginate UniProt search; cache by ``cache_name``."""
    cache = _cache_dir() / cache_name
    if cache.exists():
        data = json.loads(cache.read_text())
        logger.info("Loaded cached UniProt candidates %s (%d)", cache_name, len(data))
        return data[:n_enzymes]

    results: list[dict[str, Any]] = []
    next_url: str | None = None
    params = {
        "query": query,
        "format": "json",
        "size": 50,
        "fields": "accession,id,ec,sequence,ft_act_site,xref_pdb,protein_name",
    }
    while len(results) < n_enzymes:
        if next_url:
            resp = sess.get(next_url, timeout=120)
        else:
            resp = sess.get(UNIPROT_SEARCH, params=params, timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        results.extend(payload.get("results") or [])
        # UniProt Link header: <url>; rel="next" — URL may contain commas
        # (fields=a,b,c), so do not split the header on ",".
        link = resp.headers.get("Link") or ""
        next_url = None
        for m in re.finditer(r'<([^>]+)>\s*;\s*rel="([^"]+)"', link):
            if m.group(2) == "next":
                next_url = m.group(1)
                break
        if not next_url:
            break
        time.sleep(0.1)
    cache.write_text(json.dumps(results[:n_enzymes]))
    logger.info("Fetched %d UniProt candidates (%s)", min(len(results), n_enzymes), cache_name)
    return results[:n_enzymes]


def fetch_uniprot_catalytic_candidates(
    n_enzymes: int = 500,
    sess: requests.Session | None = None,
    include_afdb_track: bool = False,
) -> list[dict[str, Any]]:
    """Paginate UniProt for ACT_SITE proteins (PDB-first; optional AFDB-only)."""
    sess = sess or _session()
    pdb_n = n_enzymes if not include_afdb_track else max(n_enzymes - n_enzymes // 4, 1)
    af_n = n_enzymes - pdb_n if include_afdb_track else 0
    results = _paginate_uniprot_search(
        "(ft_act_site:*) AND (database:pdb)",
        pdb_n,
        sess,
        f"uniprot_act_site_pdb_n{pdb_n}.json",
    )
    if af_n > 0:
        af = _paginate_uniprot_search(
            "(ft_act_site:*) AND (reviewed:true) NOT (database:pdb)",
            af_n,
            sess,
            f"uniprot_act_site_afdb_n{af_n}.json",
        )
        seen = {r.get("primaryAccession") for r in results}
        for row in af:
            uid = row.get("primaryAccession")
            if uid and uid not in seen:
                results.append(row)
                seen.add(uid)
    return results[:n_enzymes]


def _parse_act_site_residues(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract residue numbers from UniProt active-site features.

    REST JSON uses human-readable types (``Active site``); some dumps use
    ``ACT_SITE``. Prefer catalytic active sites; ignore binding/site tags.
    """
    allowed = {"ACT_SITE", "ACTIVE SITE", "ACTIVESITE"}
    specs = []
    for feat in entry.get("features") or []:
        ftype = str(feat.get("type") or "").upper().replace("_", " ").strip()
        ftype_compact = ftype.replace(" ", "")
        if ftype not in allowed and ftype_compact not in {"ACTSITE"}:
            continue
        loc = feat.get("location") or {}
        start = (loc.get("start") or {}).get("value")
        end = (loc.get("end") or {}).get("value")
        if start is None:
            continue
        # UniProt positions are 1-indexed sequence positions — PDB mapping is
        # approximate without SIFTS; resolve via PDB auth resnums when possible.
        for pos in range(int(start), int(end or start) + 1):
            specs.append({"seq_pos": pos, "description": feat.get("description") or ""})
    return specs


def _pdb_ids_from_entry(entry: dict[str, Any]) -> list[str]:
    ids = []
    for xref in entry.get("uniProtKBCrossReferences") or []:
        if xref.get("database") == "PDB":
            pid = str(xref.get("id") or "").lower()
            if pid:
                ids.append(pid)
    return ids


def _map_seqpos_to_pdb_specs(
    pdb_text: str, seq_positions: list[int], chain: str = "A"
) -> list[dict[str, Any]]:
    """Best-effort: match CA atoms in chain order to sequence positions."""
    from catalyst_atlas.data.structures import parse_ca_atoms

    atoms = [a for a in parse_ca_atoms(pdb_text) if a.get("chain") == chain or chain == "*"]
    if not atoms:
        atoms = parse_ca_atoms(pdb_text)
    # Index by resnum when available.
    by_res = {int(a["resnum"]): a for a in atoms if a.get("resnum") is not None}
    specs = []
    for pos in seq_positions:
        a = by_res.get(int(pos))
        if a is None and 0 < pos <= len(atoms):
            a = atoms[pos - 1]
        if a is None:
            continue
        specs.append(
            {
                "chain": a.get("chain") or "A",
                "resnum": int(a["resnum"]),
                "aa": a.get("aa") or "X",
            }
        )
    return specs


def build_uniprot_atlas_rows(
    n_enzymes: int = 200,
    seed: int = 7,
    allow_alphafold: bool = True,
) -> pd.DataFrame:
    """Build atlas rows from UniProt ACT_SITE annotations."""
    import numpy as np

    rng = np.random.default_rng(seed)
    sess = _session()
    candidates = fetch_uniprot_catalytic_candidates(
        n_enzymes=max(n_enzymes * 3, 100),
        sess=sess,
        include_afdb_track=allow_alphafold,
    )
    rng.shuffle(candidates)
    pdb_cache = RAW / "pdb"
    pdb_cache.mkdir(parents=True, exist_ok=True)
    # Reserve ~25% of extras for AlphaFold-only so PDB rows do not crowd them out.
    af_quota = (n_enzymes // 4) if allow_alphafold else 0
    exp_quota = n_enzymes - af_quota
    rows: list[dict[str, Any]] = []
    n_exp = 0
    n_af = 0
    failures = 0

    def _try_row(entry: dict[str, Any], *, force_alphafold: bool) -> dict[str, Any] | None:
        nonlocal failures
        uid = entry.get("primaryAccession") or entry.get("uniProtkbId") or ""
        if not uid:
            failures += 1
            return None
        seq = ((entry.get("sequence") or {}).get("value")) or ""
        if len(seq) < 20:
            failures += 1
            return None
        act = _parse_act_site_residues(entry)
        if len(act) < 1:
            failures += 1
            return None
        seq_positions = [a["seq_pos"] for a in act][:8]
        if len(seq_positions) == 1:
            p = seq_positions[0]
            seq_positions = [max(1, p - 1), p, p + 1]
        pdb_ids = [] if force_alphafold else _pdb_ids_from_entry(entry)
        structure_source = "experimental"
        pdb_text = None
        pdb_id = None
        for pid in pdb_ids[:3]:
            pdb_text = fetch_pdb_text(pid, pdb_cache)
            if pdb_text:
                pdb_id = pid
                break
        if pdb_text is None and allow_alphafold:
            af_cache = _cache_dir() / "afdb" / f"AF-{uid}-F1-model_v4.pdb"
            af_cache.parent.mkdir(parents=True, exist_ok=True)
            if af_cache.exists():
                pdb_text = af_cache.read_text()
            else:
                try:
                    resp = sess.get(AFDB_PDB_URL.format(uid=uid), timeout=60)
                    if resp.status_code == 200 and len(resp.text) > 100:
                        af_cache.write_text(resp.text)
                        pdb_text = resp.text
                except requests.RequestException:
                    pdb_text = None
            if pdb_text:
                pdb_id = f"af-{uid.lower()}"
                structure_source = "alphafold"
        if not pdb_text or not pdb_id:
            failures += 1
            return None
        if force_alphafold and structure_source != "alphafold":
            failures += 1
            return None
        specs = _map_seqpos_to_pdb_specs(pdb_text, seq_positions)
        if len(specs) < 1:
            failures += 1
            return None
        try:
            catalytic, neighbors, ligands, cofactor_tags = build_site_from_structure(
                pdb_text, specs
            )
        except Exception:
            failures += 1
            return None
        if len(catalytic) < 1:
            failures += 1
            return None
        ec = ""
        for xref in entry.get("uniProtKBCrossReferences") or []:
            if xref.get("database") == "EC":
                ec = str(xref.get("id") or "") or ec
        pd_ = entry.get("proteinDescription") or {}
        for ecn in (pd_.get("recommendedName") or {}).get("ecNumbers") or []:
            if ecn.get("value"):
                ec = ec or str(ecn["value"])
        cat_aas = [r["aa"] for r in catalytic]
        ann = annotate_chemistry(ec_number=ec or None, catalytic_aas=cat_aas, cofactor_tags=cofactor_tags)
        ecl = ec_labels(ec)
        return {
            "enzyme_id": f"UP{uid}",
            "uniprot_id": uid,
            "pdb_id": pdb_id,
            "family_id": f"uniprot_{uid}",
            "chemistry_class": ann["chemistry_class"],
            "chemistry_family": ann["chemistry_family"],
            "mechanistic_pattern": ann["mechanistic_pattern"],
            "catalytic_pattern": "".join(cat_aas),
            "cofactor_tags": cofactor_tags,
            "substrate_class": "uniprot_act_site",
            "ec_number": ec,
            "ec_class": ecl["ec_class"],
            "ec3": ecl["ec3"],
            "sequence": seq,
            "seq_cluster": -1,
            "fold_cluster": -1,
            "site_residues_json": json.dumps(catalytic + neighbors),
            "ligands_json": json.dumps(ligands),
            "source": "uniprot",
            "structure_source": structure_source,
            "is_cryptic_seed": False,
            "cath_topology": "unknown",
        }

    seen_uids: set[str] = set()

    def _accept(row: dict[str, Any] | None) -> bool:
        if not row:
            return False
        uid = str(row["uniprot_id"])
        if uid in seen_uids:
            return False
        seen_uids.add(uid)
        rows.append(row)
        return True

    # Pass 1: fill experimental quota (prefer PDB-backed entries).
    for entry in candidates:
        if n_exp >= exp_quota:
            break
        if _pdb_ids_from_entry(entry):
            row = _try_row(entry, force_alphafold=False)
            if row and row["structure_source"] == "experimental" and _accept(row):
                n_exp += 1
                time.sleep(0.02)
    # Pass 2: fill AlphaFold quota.
    for entry in candidates:
        if n_af >= af_quota:
            break
        row = _try_row(entry, force_alphafold=True)
        if row and _accept(row):
            n_af += 1
            time.sleep(0.02)
    # Pass 3: top up remaining slots with either source.
    for entry in candidates:
        if len(rows) >= n_enzymes:
            break
        row = _try_row(entry, force_alphafold=False)
        if _accept(row):
            time.sleep(0.02)

    if not rows:
        logger.warning("UniProt expand produced zero rows (network / mapping failures)")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["fold_cluster"] = encode_cluster_ids(df["cath_topology"].tolist())
    df["seq_cluster"] = greedy_sequence_clusters(
        df["sequence"].fillna("").tolist(), threshold=0.5, metric="jaccard"
    )
    logger.info(
        "UniProt expand: %d enzymes (%d failures); experimental=%d alphafold=%d",
        len(df),
        failures,
        int((df["structure_source"] == "experimental").sum()),
        int((df["structure_source"] == "alphafold").sum()),
    )
    return df


def merge_expanded_atlas(
    base_df: pd.DataFrame,
    n_extra: int = 200,
    seed: int = 7,
    allow_alphafold: bool = True,
) -> pd.DataFrame:
    """Merge M-CSA/demo base with UniProt extras; recluster; attach EC labels."""
    base = attach_ec_labels(base_df)
    existing_up = set(base["uniprot_id"].astype(str)) if "uniprot_id" in base.columns else set()
    extra = build_uniprot_atlas_rows(
        n_enzymes=n_extra, seed=seed, allow_alphafold=allow_alphafold
    )
    if len(extra) == 0:
        logger.warning("No UniProt extras added; returning base with EC labels only")
        return base
    extra = extra[~extra["uniprot_id"].astype(str).isin(existing_up)].reset_index(drop=True)
    # Align columns
    for col in base.columns:
        if col not in extra.columns:
            extra[col] = None
    for col in extra.columns:
        if col not in base.columns:
            base[col] = None
    merged = pd.concat([base, extra[base.columns]], ignore_index=True)
    # Recluster on full set so holdouts stay consistent.
    if "sequence" in merged.columns:
        merged["seq_cluster"] = greedy_sequence_clusters(
            merged["sequence"].fillna("").tolist(), threshold=0.5, metric="jaccard"
        )
    if "cath_topology" in merged.columns:
        merged["fold_cluster"] = encode_cluster_ids(
            merged["cath_topology"].fillna("unknown").tolist()
        )
    merged = attach_ec_labels(merged)
    logger.info(
        "Expanded atlas: %d total (base=%d extra=%d); sources=%s",
        len(merged),
        len(base),
        len(extra),
        merged["source"].value_counts().to_dict(),
    )
    return merged
