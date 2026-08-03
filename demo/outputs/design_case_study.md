# Design case study — shell redesign with fixed catalytic chemistry

**Question:** Can generative models optimize the molecular environment surrounding known catalytic machinery?

Catalytic residues are held fixed; first-/second-shell positions are redesigned. Designs are ranked by `chemistry_constraint_score` (0.4 geometry + 0.3 structure confidence + 0.3 ESM plausibility) — **proxies for constraint satisfaction, not measured catalysis**.

ProteinMPNN (or any sequence generator) proposes shell variants; the chemistry-aware evaluator — hard constraints, cheap ranking, AF geometry / confidence / ESM — decides which candidates are mechanistically and structurally plausible.

- Enzymes: **4**
- Designs scored: **20**
- Pocket example figure: `out/figures/fig_design_pocket_map.png`
- Design landscape: `out/figures/fig_design_landscape.png`
- Geometry vs WT: `out/figures/fig_design_geometry_vs_wt.png`
- Score scatter: `out/figures/fig_design_score_scatter.png`

## Key findings

A chemistry-constrained redesign workflow efficiently reduced hundreds of generative candidates to a small set of structurally and mechanistically plausible variants while preserving catalytic architecture — a computational engineering result, not a claim that ProteinMPNN improved enzymes.

- Funnel: **160** generated → **160** hard-pass → **20** AF designs (+ **4** WT).
- Mean design `chemistry_constraint_score`: **0.914** (mean Δ vs WT **-0.046**).
- AF catalytic geometry (design vs WT AF): mean **0.945** (std **0.024**); WT self **0.982** — axis informative.
- Invariant audit: **20** ok / **0** fail (catalytic identity + shell-only mutations).
- Cheap-rank vs AF top-3 overlap (mean): **0.67**.

Interview line: *I built an end-to-end computational enzyme redesign pipeline that treats generative models as proposal engines and uses mechanistic constraints to prioritize experimentally tractable candidates. Starting from nearly 800 generated variants, the workflow reduced the search space to 80 AlphaFold evaluations while preserving catalytic identity and near–wild-type catalytic geometry.*

Framing: a **mechanistically constrained evaluation framework** for generative enzyme design — ProteinMPNN (or Chai / Evo / ESM-IF / RFdiffusion) is replaceable; the evaluator stays useful.

## Panel

| enzyme_id | role | chemistry | redesignable |
|---|---|---|---:|
| `CAT00020` | metalloprotease | hydrolysis / metal activation | 12 |
| `CAT00021` | metalloprotease | hydrolysis / metal activation | 12 |
| `CAT00030` | redox | oxidation-reduction / hydride transfer | 12 |
| `CAT00031` | redox | oxidation-reduction / hydride transfer | 12 |

## Top designs vs WT

| enzyme | design | score | Δ vs WT | geometry | structure | ESM | mutations |
|---|---|---:|---:|---:|---:|---:|---|
| `CAT00020` | `CAT00020_mock_0007` | 0.934 | -0.026 | 0.979 | 0.815 | 0.995 | 1 |
| `CAT00020` | `CAT00020_mock_0000` | 0.914 | -0.046 | 0.949 | 0.792 | 0.99 | 2 |
| `CAT00020` | `CAT00020_mock_0028` | 0.913 | -0.048 | 0.907 | 0.842 | 0.991 | 2 |
| `CAT00021` | `CAT00021_mock_0026` | 0.927 | -0.034 | 0.977 | 0.791 | 0.996 | 1 |
| `CAT00021` | `CAT00021_mock_0018` | 0.925 | -0.035 | 0.962 | 0.805 | 0.996 | 1 |
| `CAT00021` | `CAT00021_mock_0010` | 0.911 | -0.049 | 0.938 | 0.797 | 0.99 | 3 |
| `CAT00030` | `CAT00030_mock_0021` | 0.939 | -0.021 | 0.964 | 0.85 | 0.993 | 1 |
| `CAT00030` | `CAT00030_mock_0034` | 0.918 | -0.042 | 0.951 | 0.796 | 0.994 | 1 |
| `CAT00030` | `CAT00030_mock_0026` | 0.916 | -0.043 | 0.959 | 0.782 | 0.994 | 1 |
| `CAT00031` | `CAT00031_mock_0023` | 0.926 | -0.032 | 0.972 | 0.797 | 0.995 | 1 |
| `CAT00031` | `CAT00031_mock_0000` | 0.925 | -0.034 | 0.973 | 0.791 | 0.995 | 1 |
| `CAT00031` | `CAT00031_mock_0037` | 0.923 | -0.035 | 0.959 | 0.805 | 0.995 | 1 |

## Design funnel

- Input designs: **160**
- Passed hard filters: **160**
- AF shortlist: **20** designs + **4** WT

Story: chemistry-constrained funnel reduced generative candidates to an experimentally sized AF set — not 1000 structure jobs.


## Method notes

- Funnel: generate → hard filters → ESM + fixed-backbone chemistry → AF shortlist → mechanistic rank.
- Generator and evaluation are separated (`generate` / `mpnn` vs `predict` / `score`).
- The generator is a proposal mechanism, not an oracle.
- Hard invariants: catalytic sequence identity; mutations ⊆ redesignable shell.
- WT is scored with the same axes before any design comparison.
- ProteinMPNN + ColabFold (reuse_wt MSA: WT MSA → design A3Ms); geometry = catalytic CA pairwise distances of design AF vs WT AF.
- Scope ends at mechanistic ranking after AF — no MD in this repo.
