# Different fold, same chemistry

**Question:** Can Catalyst detect convergent chemistry across folds?


**Query enzyme:** `MCSA00176` — thermolysin
**Context:** Query fold_cluster=95; neighbors span folds [5, 14, 91, 148]

| Method | Inferred chemistry |
|---|---|
| Sequence retrieval baseline | `no confident transfer` |
| Fold / CATH retrieval baseline | `no confident transfer` |
| **Catalyst Atlas** | `hydrolysis` |

### Chemistry card
- **Chemistry family:** hydrolysis
- **Mechanistic pattern:** metal activation
- **Catalytic residues:** His-Asp-Lys
- **Cofactor / metal:** Zn
- **Confidence:** 1.00

### Evidence (top catalytic neighbors)
- `MCSA00159` — hydrolysis / metal activation (cof=Zn; d=5.183)
- `MCSA00623` — hydrolysis / metal activation (cof=Zn; d=5.249)
- `MCSA00168` — hydrolysis / metal activation (cof=Zn; d=5.263)
- `MCSA00167` — hydrolysis / metal activation (cof=Zn; d=5.547)
- `MCSA00170` — hydrolysis / metal activation (cof=Zn; d=5.587)

**Ground truth:** hydrolysis / metal activation
**Catalyst Atlas correct:** yes

**Takeaway:** Neighbors share chemistry despite different fold neighborhoods.