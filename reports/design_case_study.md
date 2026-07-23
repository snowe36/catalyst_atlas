# Design case study — shell redesign with fixed catalytic chemistry

**Question:** Can generative models optimize the molecular environment surrounding known catalytic machinery?

Catalytic residues are held fixed; first-/second-shell positions are redesigned. Designs are ranked by `chemistry_preservation_score` (0.4 geometry + 0.3 structure confidence + 0.3 ESM plausibility) — **proxies for chemistry preservation, not measured catalysis**.

- Enzymes: **8**
- Designs scored: **160**
- Pocket example figure: `reports/figures/fig_design_pocket_map.png`
- Geometry vs WT: `reports/figures/fig_design_geometry_vs_wt.png`
- Score scatter: `reports/figures/fig_design_score_scatter.png`

## Panel

| enzyme_id | role | chemistry | redesignable |
|---|---|---|---:|
| `CAT00020` | metalloprotease | hydrolysis / metal activation | 12 |
| `CAT00021` | metalloprotease | hydrolysis / metal activation | 12 |
| `CAT00030` | redox | oxidation-reduction / hydride transfer | 12 |
| `CAT00031` | redox | oxidation-reduction / hydride transfer | 12 |
| `CAT00050` | transferase | transfer / imine chemistry | 12 |
| `CAT00051` | transferase | transfer / imine chemistry | 12 |
| `CAT00080` | cofactor_dependent | elimination / metal activation | 12 |
| `CAT00081` | cofactor_dependent | elimination / metal activation | 12 |

## Top designs vs WT

| enzyme | design | score | Δ vs WT | geometry | structure | ESM | mutations |
|---|---|---:|---:|---:|---:|---:|---|
| `CAT00020` | `CAT00020_mock_0016` | 0.933 | -0.034 | 0.924 | 0.896 | 0.983 | 4 |
| `CAT00020` | `CAT00020_mock_0015` | 0.931 | -0.036 | 0.935 | 0.865 | 0.991 | 3 |
| `CAT00020` | `CAT00020_mock_0007` | 0.921 | -0.046 | 0.951 | 0.806 | 0.995 | 1 |
| `CAT00021` | `CAT00021_mock_0003` | 0.944 | -0.023 | 0.978 | 0.849 | 0.996 | 1 |
| `CAT00021` | `CAT00021_mock_0002` | 0.939 | -0.028 | 0.963 | 0.85 | 0.996 | 1 |
| `CAT00021` | `CAT00021_mock_0010` | 0.939 | -0.028 | 0.941 | 0.886 | 0.99 | 3 |
| `CAT00030` | `CAT00030_mock_0012` | 0.95 | -0.017 | 0.961 | 0.896 | 0.989 | 2 |
| `CAT00030` | `CAT00030_mock_0018` | 0.924 | -0.043 | 0.965 | 0.799 | 0.993 | 1 |
| `CAT00030` | `CAT00030_mock_0017` | 0.922 | -0.045 | 0.94 | 0.831 | 0.987 | 2 |
| `CAT00031` | `CAT00031_mock_0003` | 0.942 | -0.025 | 0.974 | 0.849 | 0.995 | 1 |
| `CAT00031` | `CAT00031_mock_0007` | 0.941 | -0.026 | 0.98 | 0.835 | 0.995 | 1 |
| `CAT00031` | `CAT00031_mock_0008` | 0.912 | -0.055 | 0.94 | 0.793 | 0.995 | 1 |
| `CAT00050` | `CAT00050_mock_0017` | 0.938 | -0.029 | 0.981 | 0.822 | 0.996 | 1 |
| `CAT00050` | `CAT00050_mock_0012` | 0.936 | -0.031 | 0.956 | 0.848 | 0.996 | 1 |
| `CAT00050` | `CAT00050_mock_0018` | 0.931 | -0.036 | 0.932 | 0.862 | 0.997 | 1 |
| `CAT00051` | `CAT00051_mock_0005` | 0.931 | -0.036 | 0.941 | 0.852 | 0.996 | 1 |
| `CAT00051` | `CAT00051_mock_0009` | 0.931 | -0.036 | 0.942 | 0.85 | 0.996 | 1 |
| `CAT00051` | `CAT00051_mock_0007` | 0.927 | -0.040 | 0.949 | 0.835 | 0.99 | 3 |
| `CAT00080` | `CAT00080_mock_0010` | 0.949 | -0.018 | 0.962 | 0.886 | 0.996 | 1 |
| `CAT00080` | `CAT00080_mock_0007` | 0.927 | -0.040 | 0.951 | 0.835 | 0.987 | 5 |
| `CAT00080` | `CAT00080_mock_0004` | 0.924 | -0.043 | 0.918 | 0.866 | 0.992 | 2 |
| `CAT00081` | `CAT00081_mock_0012` | 0.942 | -0.025 | 0.942 | 0.896 | 0.987 | 3 |
| `CAT00081` | `CAT00081_mock_0017` | 0.933 | -0.034 | 0.963 | 0.831 | 0.994 | 1 |
| `CAT00081` | `CAT00081_mock_0016` | 0.931 | -0.036 | 0.937 | 0.862 | 0.991 | 2 |

## Method notes

- Generator and evaluation are separated (`generate` / `mpnn` vs `predict` / `score`).
- Hard invariants: catalytic sequence identity; mutations ⊆ redesignable shell.
- WT is scored with the same axes before any design comparison.
- ProteinMPNN / AF2 are external runners; this report may use imported or mock predictions.
