# Catalyst Atlas — three chemistry case studies

Real M-CSA reaction centers. Not a leaderboard — three questions enzyme chemists care about.

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

---

# Case study: Different fold, same chemistry

**Question:** Can Catalyst detect convergent chemistry across folds?

# Different fold, same chemistry

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

> Representation is the **catalytic microenvironment** (reaction-center residues, cofactors/metals, geometry, first shell) — not whole-protein fold similarity or pocket shape alone.

**Takeaway:** Catalytic neighbors share chemistry family despite different fold neighborhoods — microenvironment captures convergent reaction logic.

---

# Case study: Cofactor-aware chemistry hypothesis

**Question:** Can Catalyst provide a plausible chemistry hypothesis from the reaction center?

# Cofactor-aware chemistry hypothesis

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

> Representation is the **catalytic microenvironment** (reaction-center residues, cofactors/metals, geometry, first shell) — not whole-protein fold similarity or pocket shape alone.

**Takeaway:** Cofactor/metal context in the microenvironment supports a chemistry hypothesis an enzymologist would recognize — not an EC digit alone.

---
