# Cofactor-aware chemistry hypothesis

**Question:** Can Catalyst provide a plausible chemistry hypothesis from the reaction center?


**Query enzyme:** `MCSA00661` — arylsulfatase
**Context:** Site cofactors/metals: Ca

| Method | Inferred chemistry |
|---|---|
| Sequence retrieval baseline | `no confident transfer` |
| Fold / CATH retrieval baseline | `hydrolysis` |
| **Catalyst Atlas** | `hydrolysis` |

### Chemistry card
- **Chemistry family:** hydrolysis
- **Mechanistic pattern:** metal activation
- **Catalytic residues:** Asp-Arg-His-Asn
- **Cofactor / metal:** Ca, Mg
- **Confidence:** 1.00

### Evidence (top catalytic neighbors)
- `MCSA00951` — hydrolysis / metal activation (cof=Ca; d=4.641)
- `MCSA00686` — hydrolysis / metal activation (cof=Ca; d=7.031)
- `MCSA00165` — hydrolysis / metal activation (cof=Ca; d=7.134)
- `MCSA00180` — hydrolysis / metal activation (cof=Ca,Mg; d=7.165)
- `MCSA00019` — hydrolysis / metal activation (cof=Ca; d=7.276)

**Ground truth:** hydrolysis / metal activation
**Catalyst Atlas correct:** yes

**Takeaway:** Cofactor/metal context supports a recognizable chemistry hypothesis.