# catalyst_atlas demo (5 minutes)

See the redesign funnel **without** M-CSA downloads, ProteinMPNN, or ColabFold.

## Run

```bash
# from repo root (venv recommended)
bash demo/run_demo.sh
# or:
cat-demo
```

## What you get

| Output | Meaning |
|--------|---------|
| `outputs/fig1_pipeline.png` | Structure → reaction center → chemistry |
| `outputs/fig_design_pocket_map.png` | Catalytic fixed vs redesignable shell |
| `outputs/fig_design_landscape.png` | Mock campaign score landscape |
| `outputs/design_case_study.md` | Funnel write-up (mock AF proxies) |

## Requirements

- Python ≥ 3.11
- `pip install -e ".[dev]"` (the script does this)
- CPU only

## Not in the demo

Real ProteinMPNN / ColabFold campaign artifacts are separate — mock proxies ≠ measured catalysis.
