# Catalyst Atlas — M-CSA n=959

Reaction-center representations on curated M-CSA sites. Convergent example: thermolysin ↔ neprilysin ([`hero_convergent_chemistry.md`](hero_convergent_chemistry.md)).

| Upgrade | Detail |
|---|---|
| Cofactors / metals | PDB HETATM within 8 Å of catalytic core (324 / 959 sites tagged) |
| Ontology labels | `chemistry_family` + `mechanistic_pattern` |
| External baselines | Live MMseqs2 + Foldseek chemistry-transfer (when binaries present) |
| Diagnostics | Sequence-identity stratification + fold–chemistry audits |
| Feature dim | 84 → 97 (cofactor vocabulary + coordination / ligand stats) |

## Primary result: fold-disconnected chemistry transfer

![fold-disconnected hero](figures/fig_fold_disconnected_chemistry.png)

| Method | fold_cluster accuracy |
|---|---:|
| Foldseek | 0.132 |
| MMseqs2 | 0.037 |
| **Catalyst microenvironment** | **0.369** |

## Full chemistry accuracy (`chemistry_family`)

| Split | Catalyst | MMseqs2 | Foldseek | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
| random | 0.422 | 0.286 | **0.500** | 0.724 | 0.535 |
| seq_cluster | 0.425 | 0.226 | **0.489** | 0.720 | 0.512 |
| fold_cluster | **0.369** | 0.037 | 0.132 | **0.672** | **0.458** |

## Chemistry transfer vs evolutionary distance

Nearest-train MMseqs2 `%id` on the random split:

| Bin | n | Catalyst | MMseqs2 | Foldseek |
|---|--:|---:|---:|---:|
| >80% | 2 | 0.50 | **1.00** | 0.00 |
| 40–80% | 16 | 0.62 | 0.69 | **0.75** |
| 20–40% | 60 | 0.50 | **0.70** | 0.68 |
| <20% | 114 | 0.35 | **0.00** | 0.38 |

![identity stratification](figures/fig_chemistry_by_seq_identity.png)

## Fold–chemistry relationship audits

| Audit | n | Catalyst | Foldseek | MMseqs2 | CATH fold |
|---|--:|---:|---:|---:|---:|
| Same fold, different chemistry (false-transfer trap) | 131 | 0.39 | **0.51** | 0.26 | 0.34 |
| Different fold, same chemistry (convergent recovery) | **26** | **0.50** | 0.04 | 0.08 | 0.00 |

![fold chemistry audits](figures/fig_fold_chemistry_audits.png)

Convergent audit is **n=26** — informative, not the primary claim. Lead metric is fold-disconnected transfer.

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

- Richer cofactor geometry (coordination shell, not just presence)
- Broader remote-homology coverage for identity bins
