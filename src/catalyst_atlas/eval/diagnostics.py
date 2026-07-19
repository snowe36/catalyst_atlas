"""Secondary analyses that sharpen the chemistry-vs-evolution story.

1. Stratify chemistry-transfer accuracy by nearest-train sequence identity
2. Audit same-fold / different-chemistry traps and different-fold / same-chemistry recovery
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from catalyst_atlas.eval.metrics import accuracy, stratified_accuracy

IDENTITY_BINS: list[tuple[str, float, float]] = [
    (">80%", 80.0, 100.01),
    ("40–80%", 40.0, 80.0),
    ("20–40%", 20.0, 40.0),
    ("<20%", 0.0, 20.0),
]


def identity_bin(pct: float) -> str:
    for name, lo, hi in IDENTITY_BINS:
        if lo <= pct < hi:
            return name
    return "<20%"


def nearest_train_sequence_identity(
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    mmseqs_hits: pd.DataFrame | None = None,
    seq_sim: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Nearest-train sequence identity (%) for each test enzyme.

    Prefers MMseqs2 ``pident`` when hit tables are available; otherwise falls
    back to 100 × max k-mer Jaccard to train (proxy, not true %id).
    """
    n_test = len(test_idx)
    ident = np.full(n_test, 0.0, dtype=float)
    source = "none"

    train_ids = set(meta.iloc[train_idx]["enzyme_id"].astype(str))
    test_ids = [str(meta.iloc[int(i)]["enzyme_id"]) for i in test_idx]
    id_to_test_pos = {eid: p for p, eid in enumerate(test_ids)}

    if mmseqs_hits is not None and not mmseqs_hits.empty and "pident" in mmseqs_hits.columns:
        source = "mmseqs_pident"
        for _, row in mmseqs_hits.iterrows():
            q, t = str(row["query"]), str(row["target"])
            if q not in id_to_test_pos or t not in train_ids:
                continue
            pid = float(row["pident"])
            pos = id_to_test_pos[q]
            if pid > ident[pos]:
                ident[pos] = pid
        # Queries with no MMseqs hit (e.g. UniProt extras) — fill from k-mer proxy.
        if seq_sim is not None and np.any(ident <= 0.0):
            for pos, ti in enumerate(test_idx):
                if ident[pos] > 0.0:
                    continue
                sims = seq_sim[int(ti), train_idx]
                if len(sims):
                    ident[pos] = float(np.max(sims) * 100.0)
            source = "mmseqs_pident+kmer_proxy"
        return ident, source

    if seq_sim is not None:
        source = "kmer_jaccard_proxy"
        for pos, ti in enumerate(test_idx):
            sims = seq_sim[int(ti), train_idx]
            ident[pos] = float(np.max(sims) * 100.0) if len(sims) else 0.0
        return ident, source

    return ident, source


def sequence_identity_stratified_transfer(
    y_true: list[str],
    method_preds: dict[str, list[str]],
    nearest_identity: np.ndarray,
) -> dict[str, Any]:
    """Per-bin chemistry accuracy for each method."""
    strata = [identity_bin(float(x)) for x in nearest_identity]
    bin_counts = {name: strata.count(name) for name, _, _ in IDENTITY_BINS}
    methods_out: dict[str, dict[str, float]] = {}
    for method, preds in method_preds.items():
        clean = ["__miss__" if p == "__unseen__" else p for p in preds]
        methods_out[method] = stratified_accuracy(y_true, clean, strata)
    return {
        "identity_source": None,  # filled by caller
        "bin_counts": bin_counts,
        "methods": methods_out,
    }


def _clean_preds(preds: list[str]) -> list[str]:
    return ["__miss__" if p == "__unseen__" else p for p in preds]


def fold_chemistry_audits(
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    method_preds: dict[str, list[str]],
    label_col: str = "chemistry_family",
) -> dict[str, Any]:
    """Biological wow-tests: same-fold traps and different-fold chemistry recovery."""
    train = meta.iloc[train_idx]
    test = meta.iloc[test_idx]
    y_true = test[label_col].astype(str).tolist()

    train_folds = set(train["fold_cluster"].tolist())
    # fold -> set of chemistries in train
    fold_chems: dict[Any, set[str]] = {}
    for fold, grp in train.groupby("fold_cluster"):
        fold_chems[fold] = set(grp[label_col].astype(str))

    same_fold_diff_chem_idx: list[int] = []
    diff_fold_same_chem_idx: list[int] = []

    train_chemistries = set(train[label_col].astype(str))

    for local_i, (_, row) in enumerate(test.iterrows()):
        fold = row["fold_cluster"]
        chem = str(row[label_col])
        train_chems_here = fold_chems.get(fold, set())
        # Same fold present in train, but train examples of that fold include
        # chemistry different from the query (trap for fold-based transfer).
        if fold in train_folds and train_chems_here and (
            chem not in train_chems_here or len(train_chems_here) > 1
        ):
            # Stricter trap: fold exists in train AND at least one train chem ≠ query chem
            if any(c != chem for c in train_chems_here):
                same_fold_diff_chem_idx.append(local_i)
        # No shared fold in train, but the true chemistry exists under other folds.
        if fold not in train_folds and chem in train_chemistries:
            diff_fold_same_chem_idx.append(local_i)

    def _subset_scores(indices: list[int]) -> dict[str, Any]:
        if not indices:
            return {"n": 0, "methods": {}}
        yt = [y_true[i] for i in indices]
        out_methods: dict[str, dict[str, float]] = {}
        for name, preds in method_preds.items():
            clean = _clean_preds(preds)
            yp = [clean[i] for i in indices]
            out_methods[name] = {"accuracy": accuracy(yt, yp)}
        return {"n": len(indices), "methods": out_methods}

    return {
        "same_fold_different_chemistry": {
            "description": (
                "Test enzymes whose fold appears in train with at least one "
                "different chemistry — false functional transfer trap."
            ),
            **_subset_scores(same_fold_diff_chem_idx),
        },
        "different_fold_same_chemistry": {
            "description": (
                "Test enzymes with no train fold neighbor, but whose chemistry "
                "exists under other folds — convergent chemistry recovery "
                "(informative hard audit; report n)."
            ),
            **_subset_scores(diff_fold_same_chem_idx),
        },
    }


def _cofactor_set(tags: Any) -> set[str]:
    raw = str(tags or "none")
    return {t.strip() for t in raw.split(",") if t.strip() and t.strip() != "none"}


def annotation_style_audits(
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    method_preds: dict[str, list[str]],
    X_full: np.ndarray,
    y_train: list[str],
    y_test: list[str],
    label_col: str = "chemistry_family",
    k: int = 5,
    seed: int = 7,
) -> dict[str, Any]:
    """Controls that ask: chemistry signal vs catalytic-site annotation style?

    Subsets (1–2) reuse existing method predictions. Controls (3–4) re-run
    kNN on corrupted / decoy engineered features for catalyst_microenvironment
    (and any method that can be re-scored from X_full).
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    from catalyst_atlas.eval.baselines import knn_transfer
    from catalyst_atlas.featurize.features import _aa_composition

    # Align catalytic AA / cofactor columns (prefer meta, else leave empty).
    cat_col = "catalytic_aas" if "catalytic_aas" in meta.columns else None
    cof_col = (
        "cofactor_names"
        if "cofactor_names" in meta.columns
        else ("cofactor_tags" if "cofactor_tags" in meta.columns else None)
    )

    # --- 1. Same catalytic-AA composition, different chemistry ---
    same_res_idx: list[int] = []
    if cat_col is not None:
        train_comp = np.stack(
            [_aa_composition(str(meta.iloc[int(i)].get(cat_col) or "")) for i in train_idx]
        )
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(train_comp)
        for local_i, ti in enumerate(test_idx):
            q = _aa_composition(str(meta.iloc[int(ti)].get(cat_col) or "")).reshape(1, -1)
            _, inds = nn.kneighbors(q)
            j = int(train_idx[int(inds[0, 0])])
            if str(meta.iloc[int(ti)][label_col]) != str(meta.iloc[j][label_col]):
                same_res_idx.append(local_i)

    # --- 2. Same cofactor/metal, different chemistry ---
    same_cof_idx: list[int] = []
    if cof_col is not None:
        train_cof_chem: dict[str, set[str]] = {}
        for i in train_idx:
            chem = str(meta.iloc[int(i)][label_col])
            for tag in _cofactor_set(meta.iloc[int(i)].get(cof_col)):
                train_cof_chem.setdefault(tag, set()).add(chem)
        for local_i, ti in enumerate(test_idx):
            chem = str(meta.iloc[int(ti)][label_col])
            tags = _cofactor_set(meta.iloc[int(ti)].get(cof_col))
            if not tags:
                continue
            # Shared cofactor that also appears with a different train chemistry.
            if any(
                tag in train_cof_chem and any(c != chem for c in train_cof_chem[tag])
                for tag in tags
            ):
                same_cof_idx.append(local_i)

    def _subset_from_preds(indices: list[int]) -> dict[str, Any]:
        if not indices:
            return {"n": 0, "methods": {}}
        yt = [y_test[i] for i in indices]
        out: dict[str, dict[str, float]] = {}
        for name, preds in method_preds.items():
            clean = _clean_preds(preds)
            out[name] = {"accuracy": accuracy(yt, [clean[i] for i in indices])}
        return {"n": len(indices), "methods": out}

    # --- 3. Shuffled first-shell block on engineered features ---
    # Feature layout: [cat_comp20 | shell_comp20 | cat_proxy8 | shell_proxy8 | ...]
    rng = np.random.default_rng(seed)
    X_shuf = np.array(X_full, copy=True, dtype=float)
    n = len(X_shuf)
    if X_shuf.shape[1] >= 56:
        shell_comp = X_shuf[:, 20:40].copy()
        shell_proxy = X_shuf[:, 48:56].copy()
        perm = rng.permutation(n)
        X_shuf[:, 20:40] = shell_comp[perm]
        X_shuf[:, 48:56] = shell_proxy[perm]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_shuf[train_idx])
    Xte = scaler.transform(X_shuf[test_idx])
    shuf_preds = knn_transfer(Xtr, y_train, Xte, k=k)
    shuf_acc = accuracy(y_test, shuf_preds)

    # --- 4. Decoy reaction centers (scrambled catalytic + geometry noise) ---
    X_decoy = np.array(X_full, copy=True, dtype=float)
    if X_decoy.shape[1] >= 56:
        # Replace catalytic composition with random AA compositions; scramble geom.
        for i in range(n):
            fake_aas = "".join(
                rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=int(rng.integers(2, 5)))
            )
            X_decoy[i, 0:20] = _aa_composition(fake_aas)
            # Geometry histogram block starts at 56: 20+20+8+8=56
            geom_end = min(56 + 11, X_decoy.shape[1])
            X_decoy[i, 56:geom_end] = rng.random(geom_end - 56)
    scaler_d = StandardScaler()
    Xtr_d = scaler_d.fit_transform(X_decoy[train_idx])
    Xte_d = scaler_d.transform(X_decoy[test_idx])
    decoy_preds = knn_transfer(Xtr_d, y_train, Xte_d, k=k)
    decoy_acc = accuracy(y_test, decoy_preds)

    # Chance floor: majority-class on test.
    from collections import Counter

    maj = Counter(y_train).most_common(1)[0][0]
    chance = sum(1 for y in y_test if y == maj) / max(len(y_test), 1)

    return {
        "same_residues_different_chemistry": {
            "description": (
                "Test enzymes whose nearest train neighbor by catalytic-AA "
                "composition has a different chemistry family — residue-identity shortcut."
            ),
            **_subset_from_preds(same_res_idx),
        },
        "same_cofactor_different_chemistry": {
            "description": (
                "Test enzymes sharing a cofactor/metal tag with train enzymes of "
                "a different chemistry — cofactor one-hot shortcut."
            ),
            **_subset_from_preds(same_cof_idx),
        },
        "shuffled_first_shell": {
            "description": (
                "Engineered features with first-shell composition/proxy blocks "
                "permuted across the catalog; catalytic core kept — shell annotation leakage."
            ),
            "n": int(len(y_test)),
            "methods": {
                "catalyst_microenvironment_shuffled_shell": {"accuracy": float(shuf_acc)}
            },
            "baseline_catalyst_accuracy": float(
                accuracy(y_test, _clean_preds(method_preds.get("catalyst_microenvironment", [])))
                if "catalyst_microenvironment" in method_preds
                else float("nan")
            ),
        },
        "decoy_reaction_centers": {
            "description": (
                "Queries with scrambled catalytic composition + noise geometry "
                "against real train — should approach chance if chemistry needs a real site."
            ),
            "n": int(len(y_test)),
            "chance_majority": float(chance),
            "methods": {
                "catalyst_microenvironment_decoy": {"accuracy": float(decoy_acc)}
            },
        },
    }
