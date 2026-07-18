from catalyst_atlas.data.ontology import chemistry_classes, families


def test_ontology_has_chemistry_classes():
    classes = chemistry_classes()
    assert "hydrolase" in classes
    assert "oxidoreductase" in classes


def test_families_have_catalytic_residues():
    fams = families()
    assert len(fams) >= 8
    for fam in fams:
        assert fam["catalytic_residues"]
        assert fam["chemistry_class"]
        assert fam["catalytic_pattern"]
