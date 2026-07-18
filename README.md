# Catalyst Atlas

**Inferring the chemistry a protein can perform from its catalytic microenvironment — beyond sequence and fold similarity.**

![Benchmark](https://img.shields.io/badge/benchmark-M--CSA%20v1%20n%3D959-0E7490)
![Version](https://img.shields.io/badge/version-0.2.0-1B2A2F)
[![CI](https://github.com/snowe36/catalyst_atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/snowe36/catalyst_atlas/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

**Current benchmark:** M-CSA v1 (**n=959** experimentally annotated catalytic sites) with RCSB coordinates.

> Engineered reaction-center representation + retrieval. Deep models are deferred until they win on hard holdouts.

---

## The question

Can we identify **what chemistry a protein can perform** from its catalytic microenvironment — not from sequence transfer, not from fold transfer, not from another EC classifier?

> *Ask a protein what chemistry it can do — not what it looks like.*

---

## What this is (and is not)

| Crowded framing | Catalyst Atlas |
|---|---|
| Predict EC number from sequence | Identify *chemistry* from the catalytic machine |
| Fold similarity search | Microenvironment → reaction logic |
| Binding affinity | What reaction can this site run? |

**Representation — chemistry-aware reaction center:**
- catalytic residues
- geometry among chemistry-participating atoms
- **cofactor / metal context**
- first-shell neighbors
- ligand contacts when present

**Method:** retrieval-augmented chemistry — find catalytic neighbors, then transfer/explain with evidence. The user sees a **chemistry card**, not a neighbor list alone.

We compare chemistry transfer against the strongest sequence and structure retrieval baselines available — not “we beat Foldseek.”

---

## Chemistry ontology (v0.2)

Labels an enzymologist would write — not EC-digit spam.

| Field | Examples |
|---|---|
| `chemistry_family` | hydrolysis, oxidation-reduction, transfer, carbon-carbon chemistry, ligation, isomerization, elimination |
| `mechanistic_pattern` | metal activation, nucleophile attack, acid/base catalysis, covalent intermediate, radical chemistry, hydride transfer, imine chemistry |
| `cofactor_tags` | NAD, NADP, FAD, FMN, PLP, heme, ATP, Zn, Fe, Mg, … |

---

## Real M-CSA results (n=959, v0.2 + cofactors)

See [`reports/mcsa_v02_n959_results.md`](reports/mcsa_v02_n959_results.md).

| Split | Catalyst microenv | Seq retrieval | Fold retrieval (CATH) |
|---|---:|---:|---:|
| random | **0.42** | 0.26 | 0.41 |
| seq_cluster | **0.41** | 0.21 | 0.44 |
| fold_cluster | **0.38** | 0.20 | 0.00 |

Cofactors/metals at the site lifted microenvironment accuracy (~+6 pts vs residue/geometry-only). Microenvironment beats sequence retrieval on all splits; CATH still leaks on non-fold holdouts.

**Case studies** (three stories, not 1000 metrics): `cat-cases` → [`reports/case_studies/`](reports/case_studies/)

1. Same fold, different chemistry  
2. Different fold, same chemistry  
3. Cofactor-aware chemistry hypothesis  

---

## Pipeline

```text
cat-download → cat-enrich → cat-sites → cat-embed → cat-eval → cat-cases → cat-search
```

```bash
git clone https://github.com/snowe36/catalyst_atlas.git && cd catalyst_atlas
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[dev]"

# Real curated sites (network + PDB cache)
cat-download --public --n-enzymes 1000
cat-enrich          # cofactors/metals + chemistry ontology
cat-sites && cat-embed && cat-eval
cat-cases           # three scientific case studies
cat-search --enzyme-id MCSA00001

# Synthetic harness (CI / offline)
cat-download --demo --n-enzymes 800
bash scripts/reproduce.sh
```

Optional external search tools (when installed): **MMseqs2** and **Foldseek** for sequence/structure retrieval baselines. Until then, k-mer Jaccard + CATH topology cluster-lookup are used. See `catalyst_atlas.eval.external_baselines`.

---

## Catalyst Atlas v0.2 milestone

**Goal:** chemistry-aware reaction-center representations.

| Status | Item |
|---|---|
| ✅ | M-CSA n~1000 baseline |
| ✅ | PDB cofactors / metals near the site |
| ✅ | Chemistry ontology (`chemistry_family` + `mechanistic_pattern`) |
| ⬜ | MMseqs2 chemistry-transfer baseline (binary optional) |
| ⬜ | Foldseek chemistry-transfer baseline (binary optional) |
| ✅ | 3 cryptic / convergent chemistry case studies |
| ✅ | Chemistry cards CLI output |

Deep learning is **not** the next step. A GNN that gains 2% and loses interpretability does not automatically win.

---

## Evaluation (leakage-aware)

| Split | What it tests |
|---|---|
| `random` | Optimistic ceiling |
| `seq_cluster` | Chemistry ID when sequence neighborhoods are held out |
| `fold_cluster` | Chemistry ID when fold neighborhoods are held out |

Baselines:
- **Catalyst microenvironment** kNN transfer
- **Composition-only** ablation
- **Sequence similarity** (k-mer Jaccard nearest neighbor)
- **Seq / fold cluster-lookup** (CATH for folds)
- **MMseqs2 / Foldseek** when installed

---

## Portfolio story

| Repo | Signal |
|---|---|
| `abx_atlas` | Rigorous ML pipelines for drug-discovery evaluation |
| `bgc_atlas` | Biological discovery spaces |
| **`catalyst_atlas`** | What chemistry can this protein perform? |

The artifact is not “I trained a model.” It is a system that tries to answer a question structural biologists and enzyme engineers care about.

---

## Project layout

```text
catalyst_atlas/
  src/catalyst_atlas/
    data/         # M-CSA ingest, cofactors, ontology labels, demo atlas
    site/         # microenvironment extraction
    featurize/    # chemistry + geometry + cofactor features
    models/       # engineered embed + retrieval readout
    eval/         # leakage splits, baselines, external tools
    explain/      # chemistry cards
    case_studies.py
    viz/          # offline microenvironment figures
  scripts/reproduce.sh
  tests/
  reports/
```

---

## License

MIT — see [LICENSE](LICENSE).
