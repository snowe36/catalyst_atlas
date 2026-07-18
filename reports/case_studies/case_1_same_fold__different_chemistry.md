# Case study: Same fold, different chemistry

**Question:** Can Catalyst distinguish chemistry within a structural family?

# Same fold, different chemistry

**Query enzyme:** `MCSA00034` — catechol 2,3-dioxygenase
**Context:** Shared fold cluster 19 (CATH topology neighborhood)

| Method | Inferred chemistry |
|---|---|
| Sequence retrieval baseline | `no confident transfer` |
| Fold / CATH retrieval baseline | `elimination` |
| **Catalyst Atlas** | `oxidation-reduction` |

### Chemistry card
- **Chemistry family:** oxidation-reduction
- **Mechanistic pattern:** metal activation
- **Catalytic residues:** His-Asp-Asp-His-His
- **Cofactor / metal:** Fe
- **Confidence:** 1.00

### Evidence (top catalytic neighbors)
- `MCSA00130` — oxidation-reduction / metal activation (cof=Fe; d=6.927)
- `MCSA00685` — oxidation-reduction / metal activation (cof=Fe; d=7.039)
- `MCSA00129` — oxidation-reduction / metal activation (cof=Fe; d=7.156)
- `MCSA00134` — oxidation-reduction / metal activation (cof=Fe; d=7.282)
- `MCSA00672` — oxidation-reduction / metal activation (cof=Fe; d=7.826)

**Ground truth:** oxidation-reduction / metal activation
**Catalyst Atlas correct:** yes

> Representation is the **catalytic microenvironment** (reaction-center residues, cofactors/metals, geometry, first shell) — not whole-protein fold similarity or pocket shape alone.

**Takeaway:** Fold neighborhood mixes chemistries; catalytic microenvironment recovers the reaction-center chemistry.