# Catalyst Atlas v0.2 — M-CSA n=959

Chemistry-aware reaction-center representations on curated M-CSA sites.

## What changed vs residue-only baseline

| Upgrade | Detail |
|---|---|
| Cofactors / metals | PDB HETATM within 8 Å of catalytic core (324 / 959 sites tagged) |
| Ontology labels | `chemistry_family` + `mechanistic_pattern` (not EC-digit spam) |
| Feature dim | 84 → 91 (expanded cofactor vocabulary) |

Top site tags: Mg, Zn, Mn, Ca, PLP, Fe, heme, ATP+Mg, FMN, …

## Chemistry accuracy (`chemistry_family`)

| Split | Catalyst microenv | Seq retrieval (k-mer NN) | Seq cluster-lookup | Fold retrieval (CATH) |
|---|---:|---:|---:|---:|
| random | **0.422** | 0.255 | 0.036 | 0.411 |
| seq_cluster | **0.414** | 0.210 | 0.000 | 0.441 |
| fold_cluster | **0.375** | 0.202 | 0.004 | 0.000 |

Majority-class floor ≈ 0.32 (hydrolysis). Scaler fit on train only.

## Case studies

Generated with `cat-cases` → [`reports/case_studies/`](case_studies/):

1. **Same fold, different chemistry** — `MCSA00034` (oxidation-reduction, Fe)
2. **Different fold, same chemistry** — `MCSA00176` (hydrolysis, Zn)
3. **Cofactor-aware hypothesis** — `MCSA00661` (hydrolysis, Ca)

## Reproduce

```bash
cat-download --public --n-enzymes 1000
cat-enrich
cat-sites && cat-embed && cat-eval
cat-cases
```

## Still open for v0.2+

- Wire MMseqs2 / Foldseek hit tables into chemistry-transfer baselines when binaries are present
- Richer cofactor geometry (coordination shell, not just presence)
- Keep deep models deferred until they beat this engineered baseline on hard holdouts
