# catalyst_atlas

**A leakage-aware benchmark for chemistry identification from catalytic microenvironments** — ask what chemistry a protein site supports from its reaction center, then stress-test whether that signal survives when sequence and fold neighbors are held out.

> *Catalyst Atlas evaluates whether catalytic chemistry can be identified beyond evolutionary neighborhood shortcuts.*

[![CI](https://github.com/snowe36/catalyst_atlas/actions/workflows/ci.yml/badge.svg)](https://github.com/snowe36/catalyst_atlas/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Benchmark](https://img.shields.io/badge/benchmark-expanded%20n%3D1157-0E7490)

Repo: [github.com/snowe36/catalyst_atlas](https://github.com/snowe36/catalyst_atlas)

---

## The problem

Evolutionary similarity is a strong proxy for function when homologs exist. The harder question is what remains when they do not:

**Can we tell what chemistry a site is set up to do from its microenvironment — and how much of that score is just leftover sequence/fold leakage?**

The benchmark uses sequence-cluster and fold-cluster holdouts as negative controls. This is not “beat Foldseek.” Foldseek is the right tool when fold neighbors exist. Local catalytic representations matter when chemistry is shared across unrelated folds.

---

## The key experiment: remove evolutionary neighborhoods

**Can a model identify catalytic chemistry when sequence/fold neighbors are removed?**

Primary metric: `fold_cluster` chemistry accuracy on the expanded atlas (**n=1157**; M-CSA 959 + UniProt 198).

Multi-seed bake-off (seeds 7 / 11 / 13; same protocol, different fold holdouts):

| Method | fold_cluster (mean ± std) |
|--------|--------------------------:|
| Engineered microenvironment | 0.40 ± 0.02 |
| ESM-2 | 0.40 ± 0.06 |
| ESM+GNN | 0.42 ± 0.06 |

Neighborhood baselines on a single fold holdout (seed 7) for context: MMseqs2 0.04, Foldseek 0.13.

Seed 7 example split (not the multi-seed mean):

| Method | fold_cluster accuracy |
|--------|----------------------:|
| Engineered catalytic microenvironment | 0.38 |
| ESM-2 (frozen sequence) | 0.46 |
| ESM+GNN | 0.49 |

Averaged across seeds the ESM+GNN edge over ESM-2 is small (+0.02) and sits inside the noise. Frozen protein language models already carry most fold-disconnected signal at this scale; reaction-center fusion is a complementary contribution, not a replacement.

Random-graph ablation (seed 7): ESM + **shuffled** node features scores **0.50**, matching or beating catalytic graphs (0.49). Geometry-specific gains are **not yet established** — treat the seed-7 bump as a fusion/capacity effect until a stricter graph control wins.

| Representation | fold_cluster | Reading |
|----------------|-------------:|---------|
| GNN alone (reaction-center graphs) | ~0.32 | Local graphs alone are hard at n≈1000 |
| ESM-2 | 0.40 ± 0.06 (0.46 on seed 7) | Sequence models already encode a lot of biochemistry |
| ESM+GNN | 0.42 ± 0.06 (0.49 on seed 7) | Complementary; geometry-specific gains not yet established |

Sources: [`reports/v05_seed_summary.json`](reports/v05_seed_summary.json), [`reports/v05_ablation_summary.json`](reports/v05_ablation_summary.json). Prior MCSA-only track (n=959): ESM+GNN 0.45 — see [`docs/plans/v0.3_learn_catalytic_language.md`](docs/plans/v0.3_learn_catalytic_language.md).

On a small convergent-chemistry audit (**n=29**), ESM+GNN retrieves cross-fold chemical analogs at 0.83. That subset is hypothesis-generating, not the primary benchmark.

Annotation-style controls (same-residue / same-cofactor / shuffled shell / decoy centers) are in `cat-eval` — figure [`reports/figures/fig_annotation_style_controls.png`](reports/figures/fig_annotation_style_controls.png).

<p align="center">
  <img src="reports/figures/fig_fold_disconnected_chemistry.png" alt="Fold-disconnected chemistry transfer: multi-seed means with seed-7 example split" width="720"/>
</p>

<p align="center">
  <img src="reports/figures/fig_retrieval_neighbors.png" alt="Query enzyme with ESM-2 nearest neighbor vs ESM+GNN nearest neighbor" width="720"/>
</p>

<p align="center"><em>Retrieval example: the model sometimes finds chemistry across folds — without claiming a consistent win over ESM.</em></p>

---

## Case study: convergent chemistry across unrelated folds

A fold-disconnected pair with shared chemistry recovered by reaction-center similarity — what the representation encodes, not a blind discovery claim.

| | Thermolysin `MCSA00176` | Neprilysin `MCSA00623` |
|--|-------------------------|-------------------------|
| Sequence neighborhood | remote (~5–7% k-mer Jaccard) | remote |
| Fold / CATH | `1.10.390` | `3.40.390` |
| Reaction chemistry | hydrolysis / metal activation | hydrolysis / metal activation |
| Catalyst Atlas | ranks neprilysin among top catalytic neighbors | — |

**Why:** shared reaction-center geometry · Zn cofactor environment · His/Asp/Glu catalytic arrangement — not fold TM-score.

Full writeup: [`reports/hero_convergent_chemistry.md`](reports/hero_convergent_chemistry.md).

---

## What this repo builds

1. **Download & curate** M-CSA catalytic sites with RCSB coordinates
2. **Extract** catalytic microenvironments (residues, geometry, cofactors, first shell)
3. **Represent** chemistry with engineered reaction-center features
4. **Retrieve** catalytic neighbors under leakage-aware splits
5. **Explain** with evidence cards — prediction + mechanistic evidence, not a score alone

<p align="center">
  <img src="reports/figures/fig1_pipeline.png" alt="Pipeline: structure → reaction center extraction → chemical representation → chemistry retrieval → evidence card" width="900"/>
</p>

<p align="center"><em>Figure 1. From structure to chemistry evidence — microenvironment retrieval, not fold search.</em></p>

### Example output

```bash
cat-search --enzyme-id MCSA00176
```

```text
Catalyst Atlas
==============

Chemistry: hydrolysis (metal activation)
Confidence: 0.82

Evidence:
  - catalytic residue pattern: His-Glu-Asp-Arg
  - mechanistic pattern: metal activation
  - Zn cofactor/metal at reaction center
  - analogs span multiple fold neighborhoods

Nearest analogs:
  1. MCSA00623 — hydrolysis / metal activation (cof=Zn; different fold)
  2. MCSA00159 — hydrolysis / metal activation (cof=Zn; different fold)
  ...
```

The artifact is **prediction + why** — chemistry family, mechanistic pattern, and catalytic evidence — not `score = 0.82`.

---

## Key results

| Check | Result |
|-------|--------|
| Expanded atlas sites (M-CSA + UniProt) | **1157** |
| Fold-disconnected Catalyst / MMseqs / Foldseek | **0.37** / 0.04 / 0.13 |
| Chemistry Recall@5 (fold_cluster) | **0.67** |
| Chemistry MRR (fold_cluster) | **0.46** |
| MMseqs2 at nearest-train identity **<20%** | **0.00** |
| Different-fold / same-chemistry | Catalyst **0.50** vs Foldseek **0.04** — **informative audit, n=26** |
| Same-fold / different-chemistry | Foldseek **0.51** vs Catalyst 0.39 (**n=131**) — fold info is legitimately useful |

> **Key observation:** when enzymes share chemistry but not fold, standard sequence/structure retrieval can fail; interpretable reaction-center representations provide a complementary signal.

The fold-disconnected benchmark (**n=461** test) carries the quantitative claim. The convergent-chemistry subset (**n=26**) is a biologically informative hard audit — not the primary win metric. Do not oversell it.

Full writeup: [`reports/mcsa_v02_n959_results.md`](reports/mcsa_v02_n959_results.md).

---

## Quick start

Requires **Python 3.11+**:

```bash
git clone https://github.com/snowe36/catalyst_atlas.git && cd catalyst_atlas
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[dev]"
bash scripts/reproduce.sh && pytest -q
```

```text
cat-download → cat-enrich → cat-sites → cat-embed → cat-eval → cat-cases → cat-figures
```

Optional: **MMseqs2** / **Foldseek** on `PATH` (or vendored under `tools/`) for live retrieval baselines.

---

## Figure 2 — Chemistry transfer under evolutionary distance

At high identity, sequence is the right tool. At remote homology, sequence transfer becomes unreliable; the catalytic environment retains chemically relevant signal.

| Nearest train identity | n | Catalyst | MMseqs2 | Foldseek |
|------------------------|--:|---------:|--------:|---------:|
| >80% | 3 | 0.67 | 0.33 | 0.67 |
| 40–80% | 14 | 0.50 | **0.86** | 0.71 |
| 20–40% | 61 | 0.56 | 0.64 | **0.67** |
| <20% | 154 | **0.45** | 0.00 | 0.27 |

<p align="center">
  <img src="reports/figures/fig_chemistry_by_seq_identity.png" alt="Chemistry transfer accuracy stratified by nearest-train sequence identity" width="720"/>
</p>

<p align="center"><em>Figure 2. Expanded atlas (n=1157, random split). MMseqs2 is strong in the mid-identity band and collapses below 20%; engineered microenvironments keep signal when homologs are gone. High-identity bins are small-n.</em></p>

Leakage-aware splits (expanded atlas):

| Split | Catalyst | ESM-2 | ESM+GNN | MMseqs2 | Foldseek |
|-------|---------:|------:|--------:|--------:|---------:|
| Random | 0.48 | 0.60 | **0.79** | 0.22 | 0.41 |
| Seq cluster | 0.45 | 0.59 | **0.77** | 0.22 | 0.40 |
| Fold cluster (seed 7) | 0.38 | 0.46 | **0.49** | 0.04 | 0.13 |

Fold-cluster numbers above are the seed-7 run. Multi-seed means: engineered 0.40 ± 0.02, ESM-2 0.40 ± 0.06, ESM+GNN 0.42 ± 0.06.

Recall@5 / MRR ask: does the true chemistry appear among retrieved catalytic neighbors? Accuracy alone understates a retrieval system.

---

## Figure 3 — Fold and chemistry are separable

| Panel | Question | n | Catalyst | Foldseek | MMseqs2 |
|-------|----------|--:|---------:|---------:|--------:|
| **A** Same fold, different chemistry | Avoid false functional transfer? | 131 | 0.39 | **0.51** | 0.26 |
| **B** Different fold, same chemistry | Recognize convergent chemistry? | **26** | **0.50** | 0.04 | 0.08 |

<p align="center">
  <img src="reports/figures/fig_fold_chemistry_audits.png" alt="Same-fold different-chemistry trap vs different-fold same-chemistry recovery" width="720"/>
</p>

<p align="center"><em>Figure 3. Panel A: fold information is legitimately useful — leave it visible. Panel B (n=26): chemistry can be conserved despite different evolutionary solutions.</em></p>

---

## Figure 4 — Evidence cards

The practical output is a **chemistry card**: family, mechanistic pattern, catalytic evidence, and nearest chemical analogs — not just a leaderboard number.

<p align="center">
  <img src="reports/figures/fig4_chemistry_cards.png" alt="Example chemistry cards for convergent chemistry and same-fold different-chemistry cases" width="900"/>
</p>

<p align="center"><em>Figure 4. Prediction + mechanistic evidence — the thing a scientist would actually use.</em></p>

Narrative case studies: `cat-cases` → [`reports/case_studies/`](reports/case_studies/).

---

## Catalytic microenvironment

| Component | Detail |
|-----------|--------|
| Source | [M-CSA](https://www.ebi.ac.uk/thornton-srv/m-csa/) + [RCSB PDB](https://www.rcsb.org/) |
| Catalytic residues | Annotated chemistry-participating amino acids |
| Geometry | Pairwise distances among catalytic atoms |
| Cofactors / metals | HETATM within ~8 Å of the catalytic core |
| First shell | Neighboring residues around the site |
| Labels | `chemistry_family` + `mechanistic_pattern` |

<p align="center">
  <img src="reports/figures/fig_microenv_zn_activation.png" alt="Zn-activation catalytic microenvironment" width="640"/>
</p>

<p align="center"><em>Zn-activation reaction center — a local chemical machine, not a fold fingerprint.</em></p>

---

## Data

| Item | Detail |
|------|--------|
| M-CSA curated sites | **959** |
| Additional UniProt ACT_SITE sites | **198** |
| Expanded atlas sites | **1157** |
| Structures | [RCSB PDB](https://www.rcsb.org/) + AFDB (`structure_source=alphafold` where used) |
| With site cofactors / metals (M-CSA track) | **324** |
| Convergent-chemistry audit subset | **26–29** (hard; informative, not large) |
| Demo atlas | CI harness (`cat-download --demo`) |

---

## Limitations

- Expanded atlas is still curated-scale (n=1157), not proteome-wide
- Fold-cluster scores are **split-sensitive** (ESM-2 and ESM+GNN both move ±0.06 across three seeds)
- Random-graph ablation: geometry-specific gains are **not yet established** (shuffled nodes ≥ catalytic graphs on seed 7)
- Convergent-chemistry audit is small (**n≈26–29**) — useful, not decisive
- Labels are ontology families / patterns, not full kinetic schemes
- Chemistry cards cite pattern / cofactor / fold-span evidence — not full mechanistic checklists

---

## Versions / thesis

Catalyst Atlas evaluates whether catalytic chemistry can be identified beyond evolutionary neighborhood shortcuts. At n=1157, frozen protein language models capture most fold-disconnected signal, while reaction-center representations provide a complementary but not yet fully isolated contribution. The benchmark emphasizes leakage-aware evaluation and mechanistic evidence rather than raw prediction scores.

| Version | Claim |
|---------|--------|
| v0.2 | Catalytic microenvironments contain chemistry signal under fold holdout |
| v0.3 | Learned representations require leakage-aware evaluation |
| v0.4 | Expanded atlas + controls separate chemistry from annotation shortcuts |
| v0.5 | ESM+GNN can improve a fold holdout on individual splits (seed 7: 0.49), but multi-seed and random-graph controls show that geometry-specific gains are not yet established |

| Step | Command | Role |
|------|---------|------|
| Reaction-center graphs | `cat-graphs` | Explicit catalytic-machine graphs |
| Frozen ESM-2 | `cat-esm` | Sequence foundation-model control |
| ESM + GNN fusion | `cat-train-encoder --fusion-esm` | Sequence + local structure |
| Annotation controls | `cat-eval` | Same-residue / same-cofactor / shuffled shell / decoy |
| Expanded atlas | `cat-download --public --expanded` | UniProt ACT_SITE + EC; AFDB as `structure_source=alphafold` |
| Multi-seed bake-off | `scripts/v05_seed_bakeoff.py` / `v05_parallel_bakeoff.py` | Split variance |
| Random-graph ablation | `scripts/v05_ablation_run.py` | Geometry vs capacity control |

Lead metric: **`fold_cluster`**. Summaries: [`v04_reeval`](reports/v04_reeval_summary.json), [`v05_seed`](reports/v05_seed_summary.json), [`v05_ablation`](reports/v05_ablation_summary.json). Plans: [`v0.3`](docs/plans/v0.3_learn_catalytic_language.md), [`v0.4`](docs/plans/v0.4_rigor_and_scale.md), [`v0.5`](docs/plans/v0.5_expanded_learned.md).

Out of scope for now: ESM fine-tuning, hard-negative mining, multi-task heads, optimizing random-split accuracy.

---

## How to reproduce

```bash
git clone https://github.com/snowe36/catalyst_atlas.git
cd catalyst_atlas
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[dev]"
bash scripts/reproduce.sh
pytest -q
```

Real curated sites:

```bash
cat-download --public --n-enzymes 1000
cat-enrich
cat-sites && cat-embed && cat-eval
cat-cases && cat-figures
cat-search --enzyme-id MCSA00176
```

| Artifact | Path |
|----------|------|
| Fig 1 pipeline | `reports/figures/fig1_pipeline.png` |
| Fig 2 identity stratification | `reports/figures/fig_chemistry_by_seq_identity.png` |
| Fig 3 fold–chemistry audits | `reports/figures/fig_fold_chemistry_audits.png` |
| Fig 4 chemistry cards | `reports/figures/fig4_chemistry_cards.png` |
| Retrieval neighbors | `reports/figures/fig_retrieval_neighbors.png` |
| Annotation controls | `reports/figures/fig_annotation_style_controls.png` |
| Convergent case study | `reports/hero_convergent_chemistry.md` |
| Metrics | `data/processed/eval_metrics.json` |
| Seed / ablation summaries | `reports/v05_seed_summary.json`, `reports/v05_ablation_summary.json` |

---

## Project layout

```text
src/catalyst_atlas/   package (data, site, featurize, models, eval, explain, viz)
scripts/              reproduce.sh, embed_esm.py, runpod_train.sh
docs/plans/           v0.3 bake-off + v0.4 rigor/scale plans
data/raw|processed/   M-CSA / PDB cache + features, graphs, embeddings, metrics
artifacts/            trained encoder checkpoints (gitignored)
reports/figures/      Fig 1–4 + microenvironment panels
reports/case_studies/ three scientific narratives
tests/                unit + pipeline smoke tests
.github/workflows/    CI (ruff + pytest; GPU optional)
```

---

## Acknowledgments

Catalytic site annotations from [M-CSA](https://www.ebi.ac.uk/thornton-srv/m-csa/) (EMBL-EBI / Thornton group). Structures from the [RCSB PDB](https://www.rcsb.org/).

---

## License

MIT
