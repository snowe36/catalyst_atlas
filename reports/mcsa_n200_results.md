# M-CSA real-data run (n≈196)

First end-to-end run on curated [M-CSA](https://www.ebi.ac.uk/thornton-srv/m-csa/) catalytic sites with RCSB coordinates — **not** the synthetic demo atlas.

## Setup

```bash
cat-download --public --n-enzymes 200
cat-sites && cat-embed && cat-eval
```

| Field | Value |
|---|---|
| Source | M-CSA API + RCSB PDB + UniProt |
| Enzymes retained | 196 / 200 (4 PDB/map failures) |
| Chemistry classes | oxidoreductase 58, hydrolase 51, transferase 35, lyase 24, isomerase 19, ligase 7, translocase 2 |
| Seq clusters | 186 (k-mer containment ≥ 0.35) |
| Fold clusters | 107 (CATH topology) |
| Scaler | `StandardScaler` fit on train split only |

## Results (chemistry-class accuracy)

| Split | Catalyst microenvironment | Seq similarity (k-mer) | Seq cluster-lookup | Fold cluster-lookup (CATH) |
|---|---:|---:|---:|---:|
| random | **0.35** | 0.15 | 0.00 | 0.25 |
| seq_cluster | **0.28** | 0.18 | 0.00 | 0.21 |
| fold_cluster | 0.23 | **0.31** | 0.00 | 0.00 |

Majority-class floor ≈ 0.30 (oxidoreductase). Metrics land in `data/processed/eval_metrics.json`.

## Interpretation

- On **real** sites the problem is hard — far from the synthetic 0.99 ceiling.
- Microenvironment kNN beats k-mer sequence transfer on random / seq-cluster holdouts, but **loses on fold holdout** in this first slice.
- Seq cluster-lookup is near-zero because M-CSA entries are mostly unique mechanisms (few multi-member sequence neighborhoods at this threshold).
- Fold clusters are real CATH topologies; holding them out is a meaningful leakage control.

## Next

1. Scale toward full M-CSA (~1000 entries).
2. Wire mmseqs2 / Foldseek when available for stronger sequence/fold baselines.
3. Add cofactor ligands from M-CSA / PDB HETATM near the site.
4. Only then consider deeper models — they must win on these hard holdouts.
