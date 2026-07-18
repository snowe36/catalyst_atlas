# M-CSA real-data run (n=959)

Full curated [M-CSA](https://www.ebi.ac.uk/thornton-srv/m-csa/) ingest with RCSB coordinates — not the synthetic demo.

## Setup

```bash
cat-download --public --n-enzymes 1000
cat-sites && cat-embed && cat-eval
```

| Field | Value |
|---|---|
| Source | M-CSA API + RCSB PDB + UniProt |
| Enzymes retained | 959 / 1000 |
| Chemistry classes | hydrolase 303, oxidoreductase 202, transferase 192, lyase 134, isomerase 79, ligase 39, translocase 10 |
| Seq clusters | 926 (k-mer Jaccard ≥ 0.15; mid-length seeds) |
| Fold clusters | 246 (CATH topology) |
| Majority-class floor | ≈0.32 (hydrolase) |
| Scaler | `StandardScaler` fit on train split only |

## Results (chemistry-class accuracy)

| Split | Catalyst microenvironment | Seq similarity (k-mer Jaccard) | Seq cluster-lookup | Fold cluster-lookup (CATH) |
|---|---:|---:|---:|---:|
| random | **0.36** | 0.26 | 0.03 | 0.45 |
| seq_cluster | **0.35** | 0.21 | 0.00 | 0.44 |
| fold_cluster | **0.31** | 0.20 | 0.00 | 0.00 |

Metrics: `data/processed/eval_metrics.json`. Figure: `reports/figures/fig_chemistry_leakage.png`.

## Interpretation

- Real M-CSA is hard: ~0.35 accuracy vs synthetic demo ~0.99.
- Microenvironment kNN beats k-mer sequence transfer on all three splits.
- CATH fold-lookup is strong when fold neighborhoods leak (~0.44–0.45) and collapses to 0 on fold holdout — as designed.
- M-CSA entries are mostly unique mechanisms, so sequence cluster-lookup stays near zero; the informative sequence baseline is **nearest-neighbor k-mer similarity**.

## Notes / caveats

- Cofactors are not yet parsed from M-CSA/PDB (`cofactor_tags=none`).
- Sequence baseline is k-mer Jaccard, not mmseqs2/BLAST %id.
- Fold baseline is CATH topology cluster-lookup, not Foldseek TM-score search.
