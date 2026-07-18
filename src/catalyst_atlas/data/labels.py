"""Chemistry ontology labels — enzymologist-facing, not EC-digit spam."""

from __future__ import annotations

from typing import Any

# Broad reaction logic (primary retrieval / eval label).
CHEMISTRY_FAMILIES = [
    "hydrolysis",
    "oxidation-reduction",
    "transfer",
    "carbon-carbon chemistry",
    "ligation",
    "isomerization",
    "elimination",
    "unknown",
]

# How the site does chemistry (secondary card field).
MECHANISTIC_PATTERNS = [
    "metal activation",
    "nucleophile attack",
    "acid/base catalysis",
    "covalent intermediate",
    "radical chemistry",
    "hydride transfer",
    "imine chemistry",
    "unknown",
]

_EC_TO_FAMILY = {
    "1": "oxidation-reduction",
    "2": "transfer",
    "3": "hydrolysis",
    "4": "elimination",  # lyases — bond cleavage / elimination-ish
    "5": "isomerization",
    "6": "ligation",
    "7": "transfer",  # translocases: not chemistry-first; park under transfer
}

# Legacy EC-class names used by the synthetic demo → family
_CLASS_TO_FAMILY = {
    "oxidoreductase": "oxidation-reduction",
    "transferase": "transfer",
    "hydrolase": "hydrolysis",
    "lyase": "elimination",
    "isomerase": "isomerization",
    "ligase": "ligation",
    "translocase": "transfer",
}

_METALS = {"Zn", "Fe", "Mg", "Mn", "Ca", "Cu", "Co", "Ni"}


def chemistry_family_from_ec(ec: str | None) -> str:
    if not ec:
        return "unknown"
    return _EC_TO_FAMILY.get(str(ec).split(".")[0], "unknown")


def chemistry_family_from_class(chemistry_class: str | None) -> str:
    if not chemistry_class:
        return "unknown"
    return _CLASS_TO_FAMILY.get(str(chemistry_class), "unknown")


def refine_family_for_lyase(ec: str | None, family: str) -> str:
    """Some EC 4.* reactions are better described as carbon-carbon chemistry."""
    if family != "elimination" or not ec:
        return family
    parts = str(ec).split(".")
    # 4.1.* carbon-carbon lyases
    if len(parts) >= 2 and parts[0] == "4" and parts[1] == "1":
        return "carbon-carbon chemistry"
    return family


def mechanistic_pattern_from_site(
    catalytic_aas: list[str] | str,
    cofactor_tags: str | list[str] | None = None,
    chemistry_family: str | None = None,
) -> str:
    """Infer a coarse mechanistic pattern from residues + cofactors."""
    if isinstance(catalytic_aas, str):
        aas = list(catalytic_aas)
    else:
        aas = list(catalytic_aas or [])
    aa_set = set(aas)

    if isinstance(cofactor_tags, str):
        tags = {t.strip() for t in cofactor_tags.split(",") if t.strip() and t.strip() != "none"}
    else:
        tags = {str(t) for t in (cofactor_tags or []) if t and t != "none"}

    # Cofactor-driven patterns first (strong chemistry signal).
    if tags & {"heme", "FAD", "FMN"} and chemistry_family == "oxidation-reduction":
        return "radical chemistry"
    if tags & {"NAD", "NADP"}:
        return "hydride transfer"
    if "PLP" in tags:
        return "imine chemistry"
    if tags & _METALS:
        return "metal activation"

    # Residue-driven patterns.
    if aa_set & {"S", "C", "T", "K"} and aa_set & {"H", "D", "E"}:
        # Classic triad / covalent nucleophile + acid-base support
        if aa_set & {"S", "C"}:
            return "covalent intermediate"
        return "nucleophile attack"
    if aa_set & {"H", "D", "E", "Y"}:
        return "acid/base catalysis"
    if aa_set & {"S", "C", "K"}:
        return "nucleophile attack"

    return "unknown"


def annotate_chemistry(
    *,
    ec_number: str | None = None,
    chemistry_class: str | None = None,
    catalytic_aas: list[str] | str | None = None,
    cofactor_tags: str | list[str] | None = None,
) -> dict[str, Any]:
    """Return chemistry_family + mechanistic_pattern (+ legacy class mirror)."""
    if chemistry_class:
        family = chemistry_family_from_class(chemistry_class)
    else:
        family = chemistry_family_from_ec(ec_number)
    family = refine_family_for_lyase(ec_number, family)
    pattern = mechanistic_pattern_from_site(
        catalytic_aas or [],
        cofactor_tags=cofactor_tags,
        chemistry_family=family,
    )
    # Keep a short legacy class for older demo code paths.
    legacy = {
        "hydrolysis": "hydrolase",
        "oxidation-reduction": "oxidoreductase",
        "transfer": "transferase",
        "elimination": "lyase",
        "carbon-carbon chemistry": "lyase",
        "isomerization": "isomerase",
        "ligation": "ligase",
    }.get(family, chemistry_class or "unknown")
    return {
        "chemistry_family": family,
        "mechanistic_pattern": pattern,
        "chemistry_class": legacy,
    }
