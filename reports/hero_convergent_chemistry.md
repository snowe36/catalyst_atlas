# Case study: convergent chemistry across unrelated folds

A fold-disconnected pair with shared chemistry recovered by reaction-center similarity — an analysis of what the representation encodes, not a blind discovery claim.

**Query:** `MCSA00176` — thermolysin  
**Catalytic analog:** `MCSA00623` — neprilysin

| Signal | Observation |
|--------|-------------|
| Sequence neighborhood | remote (~5–7% k-mer Jaccard; no confident MMseqs transfer) |
| Fold similarity | low (distinct CATH topologies: `1.10.390` vs `3.40.390`) |
| Same reaction chemistry | **yes** — hydrolysis / metal activation |
| Catalyst Atlas | ranks neprilysin among top catalytic neighbors |

**Shared catalytic features (why the transfer):**

- ✓ Zn cofactor environment at the reaction center
- ✓ metal-activation mechanistic pattern
- ✓ His / Asp / Glu catalytic residue arrangement
- ✓ reaction-center geometry — not whole-protein fold TM-score

Sequence and structure say these enzymes are unrelated. The catalytic microenvironment places them in the same chemistry neighborhood.
