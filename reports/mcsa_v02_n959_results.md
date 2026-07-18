# Catalyst Atlas v0.2 — M-CSA n=959

Chemistry-aware reaction-center representations on curated M-CSA sites.

**Claim:** catalytic microenvironment representations retain chemistry signal when evolutionary retrieval loses access to homologous neighborhoods — not “we beat Foldseek.”

## What changed vs residue-only baseline

| Upgrade | Detail |
|---|---|
| Cofactors / metals | PDB HETATM within 8 Å of catalytic core (324 / 959 sites tagged) |
| Ontology labels | `chemistry_family` + `mechanistic_pattern` (not EC-digit spam) |
| External baselines | Live MMseqs2 + Foldseek chemistry-transfer (when binaries present) |
| Feature dim | 84 → 97 (cofactor vocabulary + coordination / ligand stats) |

Top site tags: Mg, Zn, Mn, Ca, PLP, Fe, heme, ATP+Mg, FMN, …

## Primary result: fold-disconnected chemistry transfer

![fold-disconnected hero](figures/fig_fold_disconnected_chemistry.png)

| Method | fold_cluster accuracy |
|---|---:|
| Foldseek | 0.132 |
| MMseqs2 | 0.037 |
| **Catalyst microenvironment** | **0.369** |

When fold neighborhoods are held out, sequence and structure retrieval collapse; the microenvironment retains chemistry signal. That is the designed experiment.

## Full chemistry accuracy (`chemistry_family`)

| Split | Catalyst | MMseqs2 | Foldseek | Seq (k-mer NN) | Fold (CATH) |
|---|---:|---:|---:|---:|---:|
| random | 0.422 | 0.286 | **0.500** | 0.255 | 0.411 |
| seq_cluster | 0.425 | 0.226 | **0.489** | 0.210 | 0.441 |
| fold_cluster | **0.369** | 0.037 | 0.132 | 0.202 | 0.000 |

Foldseek dominating on random / seq_cluster is expected: fold neighbors carry chemistry when available. Leaving that visible validates the evaluation framework.

Majority-class floor ≈ 0.32 (hydrolysis). Scaler fit on train only.

## Case studies

Generated with `cat-cases` → [`reports/case_studies/`](case_studies/):

1. **Same fold, different chemistry** — avoid false functional transfer
2. **Different fold, same chemistry** — convergent chemistry recognition
3. **Cofactor-aware hypothesis** — metal/cofactor context at the site

## Reproduce

```bash
cat-download --public --n-enzymes 1000
cat-enrich
cat-sites && cat-embed && cat-eval
cat-cases
```

Requires `mmseqs` / `foldseek` on `PATH` (or under `tools/{mmseqs,foldseek}/bin`) for the external columns. On Apple Silicon with a Rosetta Python, eval wraps those binaries with `arch -arm64`.

## Still open

- Stratify accuracy by nearest-neighbor sequence identity (>80 / 40–80 / 20–40 / <20)
- Same-fold / different-chemistry false-transfer audit at scale
- Richer cofactor geometry (coordination shell, not just presence)
- Keep deep models deferred until they beat this engineered baseline on hard holdouts
