# Cryptic chemistry case

> Synthetic demo atlas.

**Query enzyme:** `CAT00362`
**Sequence context:** synthetic demo (cluster-lookup placeholders, not real BLAST %id)

| Method | Inferred chemistry |
|---|---|
| Seq cluster-lookup (BLAST placeholder) | `oxidoreductase` |
| Fold cluster-lookup (Foldseek placeholder) | `hydrolase` |
| **Catalyst Atlas** | `transferase` |

### Chemistry card
- **Reaction chemistry:** transferase
- **Catalytic pattern:** PLP-imine
- **Cofactor / metal:** PLP
- **Confidence:** 1.00

### Evidence (top catalytic neighbors)
- `CAT00373` — transferase / PLP-imine (distance 2.370; seq_cluster=49)
- `CAT00367` — transferase / PLP-imine (distance 2.799; seq_cluster=43)
- `CAT00383` — transferase / PLP-imine (distance 2.840; seq_cluster=6)
- `CAT00391` — transferase / PLP-imine (distance 3.045; seq_cluster=14)
- `CAT00365` — transferase / PLP-imine (distance 3.259; seq_cluster=41)

**Ground truth:** transferase / PLP-imine
**Catalyst Atlas correct:** yes

> Representation is the **catalytic microenvironment** (chemistry residues, cofactors, catalytic geometry, ligand contacts) — not whole-protein fold similarity or pocket shape alone.