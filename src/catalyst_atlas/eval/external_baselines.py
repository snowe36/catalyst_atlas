"""Live MMseqs2 / Foldseek chemistry-transfer baselines.

Compares *chemistry capability transfer* against modern sequence/structure
retrieval — not a claim that Catalyst "beats Foldseek."

Binaries are resolved from PATH or ``tools/{mmseqs,foldseek}/bin`` (vendored
macOS builds). When unavailable, methods report ``status: unavailable`` and
eval skips them.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from catalyst_atlas.paths import PROCESSED, RAW, ROOT, ensure_dirs

logger = logging.getLogger(__name__)

# Foldseek often reports chain-qualified ids (MCSA00335_A); strip trailing chain.
_CHAIN_SUFFIX = re.compile(r"_[A-Za-z0-9]{1,2}$")


def _normalize_hit_id(raw: str) -> str:
    """Map Foldseek/MMseqs hit ids back to atlas enzyme_id keys."""
    eid = str(raw).split(".")[0]
    stem = _CHAIN_SUFFIX.sub("", eid)
    # Only strip when the stem still looks like an enzyme id (avoid eating real ids).
    if stem != eid and (stem.startswith("MCSA") or stem.startswith("D")):
        return stem
    return eid


def _repo_tool(name: str) -> str | None:
    cand = ROOT / "tools" / name / "bin" / name
    if cand.exists() and os.access(cand, os.X_OK):
        return str(cand)
    return None


def _which(name: str) -> str | None:
    return shutil.which(name) or _repo_tool(name)


def tool_status() -> dict[str, str | None]:
    return {"mmseqs": _which("mmseqs"), "foldseek": _which("foldseek")}


def write_fasta(meta: pd.DataFrame, path: Path) -> int:
    n = 0
    with path.open("w") as fh:
        for _, row in meta.iterrows():
            seq = (row.get("sequence") or "").strip()
            if len(seq) < 20:
                continue
            fh.write(f">{row['enzyme_id']}\n{seq}\n")
            n += 1
    return n


def _apple_silicon() -> bool:
    """True on Apple Silicon, even when this Python is running under Rosetta."""
    if platform.system() != "Darwin":
        return False
    if platform.machine() == "arm64":
        return True
    try:
        # 1 when hw can run arm64 (Apple Silicon), including under Rosetta.
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.optional.arm64"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return out == "1"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


def _prefer_native_arch(cmd: list[str]) -> list[str]:
    """Force arm64 on Apple Silicon.

    Universal mmseqs/foldseek binaries hang forever under Rosetta in
    ``_GLOBAL__sub_I_mmseqs.cpp`` during dyld init. A Rosetta-translated Python
    (``platform.machine() == "x86_64"``) would otherwise spawn that path —
    so we detect Apple Silicon via ``hw.optional.arm64`` and wrap with
    ``arch -arm64``.
    """
    if _apple_silicon() and shutil.which("arch") and cmd and cmd[0] != "arch":
        return ["arch", "-arm64", *cmd]
    return cmd


def _run(cmd: list[str], cwd: Path | None = None, log_path: Path | None = None) -> None:
    """Run an external tool without pipe-buffer deadlocks (mmseqs is very chatty)."""
    cmd = _prefer_native_arch(cmd)
    logger.info("Running: %s", " ".join(cmd[:6]) + ("…" if len(cmd) > 6 else ""))
    if log_path is None:
        # Discard verbose tool logs; keep process from blocking on full pipes.
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=cwd,
        )
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, cwd=cwd)


def ensure_mmseqs_hits(
    meta: pd.DataFrame,
    threads: int = 4,
    force: bool = False,
) -> Path | None:
    """All-vs-all MMseqs2 easy-search; cache hits under data/processed/."""
    ensure_dirs()
    out = PROCESSED / "mmseqs_hits.tsv"
    exe = _which("mmseqs")
    if not exe:
        logger.warning("mmseqs not found — sequence retrieval baseline unavailable")
        return None
    if out.exists() and out.stat().st_size > 0 and not force:
        logger.info("Using cached MMseqs2 hits: %s", out)
        return out

    with tempfile.TemporaryDirectory(prefix="cat_mmseqs_") as tmp:
        tmp_path = Path(tmp)
        fasta = tmp_path / "atlas.fa"
        n = write_fasta(meta, fasta)
        if n < 2:
            logger.warning("Not enough sequences for MMseqs2")
            return None
        raw_out = tmp_path / "hits.tsv"
        cmd = [
            exe,
            "easy-search",
            str(fasta),
            str(fasta),
            str(raw_out),
            str(tmp_path / "tmp"),
            "--threads",
            str(threads),
            "-e",
            "1e-3",
            "--max-seqs",
            "50",
            "--format-output",
            "query,target,pident,evalue,bits",
        ]
        try:
            _run(cmd)
        except subprocess.CalledProcessError as exc:
            logger.error("mmseqs failed: %s\n%s", exc.stderr[-500:] if exc.stderr else exc)
            return None
        if not raw_out.exists():
            return None
        # Drop self-hits
        lines = []
        for line in raw_out.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            if parts[0] == parts[1]:
                continue
            lines.append(line)
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        logger.info("Wrote %d MMseqs2 hits → %s", len(lines), out)
        return out


def ensure_foldseek_hits(
    meta: pd.DataFrame,
    threads: int = 4,
    force: bool = False,
) -> Path | None:
    """All-vs-all Foldseek easy-search over cached PDBs."""
    ensure_dirs()
    out = PROCESSED / "foldseek_hits.tsv"
    exe = _which("foldseek")
    if not exe:
        logger.warning("foldseek not found — structure retrieval baseline unavailable")
        return None
    if out.exists() and out.stat().st_size > 0 and not force:
        logger.info("Using cached Foldseek hits: %s", out)
        return out

    pdb_dir = RAW / "mcsa_cache" / "pdb"
    if not pdb_dir.exists():
        logger.warning("No PDB cache at %s", pdb_dir)
        return None

    # Embedding meta often drops pdb_id; recover from atlas / microenvironments.
    work = meta
    if "pdb_id" not in work.columns or work["pdb_id"].fillna("").astype(str).str.len().eq(0).all():
        for src in (RAW / "catalytic_atlas.parquet", PROCESSED / "microenvironments.parquet"):
            if not src.exists():
                continue
            extra = pd.read_parquet(src, columns=["enzyme_id", "pdb_id"])
            work = work.drop(columns=["pdb_id"], errors="ignore").merge(
                extra, on="enzyme_id", how="left"
            )
            if work["pdb_id"].fillna("").astype(str).str.len().gt(0).any():
                logger.info("Joined pdb_id for Foldseek from %s", src.name)
                break

    # Symlink/copy only PDBs present in meta into a flat search dir with enzyme ids.
    # Foldseek keys = filenames. Map pdb_id -> enzyme_id (first wins if duplicates).
    pdb_to_enzyme: dict[str, str] = {}
    for _, row in work.iterrows():
        pid = str(row.get("pdb_id") or "").lower()
        if pid and pid not in pdb_to_enzyme:
            pdb_to_enzyme[pid] = str(row["enzyme_id"])

    with tempfile.TemporaryDirectory(prefix="cat_foldseek_") as tmp:
        tmp_path = Path(tmp)
        struct_dir = tmp_path / "structs"
        struct_dir.mkdir()
        n_linked = 0
        for pid, eid in pdb_to_enzyme.items():
            src = pdb_dir / f"{pid}.pdb"
            if not src.exists():
                continue
            # Name file by enzyme_id so hits map directly.
            dst = struct_dir / f"{eid}.pdb"
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                dst.write_bytes(src.read_bytes())
            n_linked += 1
        if n_linked < 2:
            logger.warning("Not enough PDBs for Foldseek (%d)", n_linked)
            return None

        raw_out = tmp_path / "hits.tsv"
        cmd = [
            exe,
            "easy-search",
            str(struct_dir),
            str(struct_dir),
            str(raw_out),
            str(tmp_path / "tmp"),
            "--threads",
            str(threads),
            "-e",
            "0.01",
            "--max-seqs",
            "50",
            "--format-output",
            "query,target,evalue,bits,alntmscore",
        ]
        try:
            _run(cmd)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "foldseek failed: %s\n%s",
                exc,
                (exc.stderr or "")[-800:],
            )
            return None
        if not raw_out.exists():
            return None
        lines = []
        for line in raw_out.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            q = _normalize_hit_id(parts[0])
            t = _normalize_hit_id(parts[1])
            if q == t:
                continue
            parts[0], parts[1] = q, t
            lines.append("\t".join(parts))
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        logger.info("Wrote %d Foldseek hits → %s", len(lines), out)
        return out


def load_hits(path: Path, kind: str = "mmseqs") -> pd.DataFrame:
    """Load hit TSV. ``kind`` is ``mmseqs`` or ``foldseek``."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["query", "target", "score"])
    df = pd.read_csv(path, sep="\t", header=None)
    if kind == "foldseek" and df.shape[1] >= 5:
        # query,target,evalue,bits,alntmscore
        df = df.iloc[:, :5].copy()
        df.columns = ["query", "target", "evalue", "bits", "alntmscore"]
        df["score"] = pd.to_numeric(df["alntmscore"], errors="coerce").fillna(0.0)
    elif df.shape[1] >= 5:
        # query,target,pident,evalue,bits
        df = df.iloc[:, :5].copy()
        df.columns = ["query", "target", "pident", "evalue", "bits"]
        df["score"] = pd.to_numeric(df["bits"], errors="coerce").fillna(0.0)
    else:
        df = df.copy()
        df.columns = ["query", "target"] + [f"c{i}" for i in range(2, df.shape[1])]
        df["score"] = 1.0
    df["query"] = df["query"].astype(str).map(_normalize_hit_id)
    df["target"] = df["target"].astype(str).map(_normalize_hit_id)
    return df


def retrieval_chemistry_transfer(
    hits: pd.DataFrame,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    label_col: str = "chemistry_family",
) -> list[str]:
    """Transfer chemistry from the best retrieval hit in the train set."""
    train_ids = set(meta.iloc[train_idx]["enzyme_id"].astype(str))
    id_to_label = {
        str(r["enzyme_id"]): str(r[label_col]) for _, r in meta.iloc[train_idx].iterrows()
    }
    # Best train hit per query
    best: dict[str, tuple[float, str]] = {}
    if not hits.empty:
        for _, row in hits.iterrows():
            q, t, s = str(row["query"]), str(row["target"]), float(row["score"])
            if t not in train_ids:
                continue
            prev = best.get(q)
            if prev is None or s > prev[0]:
                best[q] = (s, t)

    preds: list[str] = []
    for i in test_idx:
        qid = str(meta.iloc[int(i)]["enzyme_id"])
        hit = best.get(qid)
        if hit is None:
            preds.append("__unseen__")
        else:
            preds.append(id_to_label.get(hit[1], "__unseen__"))
    return preds


def prepare_retrieval_baselines(
    meta: pd.DataFrame,
    threads: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    """Run/cache MMseqs2 + Foldseek and return loaded hit tables."""
    status = tool_status()
    result: dict[str, Any] = {
        "framing": (
            "Compare chemistry transfer against modern sequence and structure "
            "retrieval baselines."
        ),
        "tools": {k: v is not None for k, v in status.items()},
        "mmseqs_hits": None,
        "foldseek_hits": None,
    }
    mm_path = ensure_mmseqs_hits(meta, threads=threads, force=force)
    if mm_path is not None:
        result["mmseqs_hits"] = load_hits(mm_path, kind="mmseqs")
        result["mmseqs_n_hits"] = int(len(result["mmseqs_hits"]))
    fs_path = ensure_foldseek_hits(meta, threads=threads, force=force)
    if fs_path is not None:
        result["foldseek_hits"] = load_hits(fs_path, kind="foldseek")
        result["foldseek_n_hits"] = int(len(result["foldseek_hits"]))
    return result
