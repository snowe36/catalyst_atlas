"""Unit tests for UniProt expand helpers (no network)."""

from __future__ import annotations

import re

from catalyst_atlas.data.uniprot_expand import _parse_act_site_residues, ec_labels


def test_ec_labels():
    assert ec_labels("3.4.24.27") == {"ec_class": "3", "ec3": "3.4.24"}
    assert ec_labels("")["ec_class"] == "unknown"


def test_parse_active_site_human_readable():
    entry = {
        "features": [
            {
                "type": "Active site",
                "location": {
                    "start": {"value": 212},
                    "end": {"value": 212},
                },
            },
            {
                "type": "Active site",
                "location": {
                    "start": {"value": 248},
                    "end": {"value": 248},
                },
            },
            {
                "type": "Binding site",
                "location": {"start": {"value": 10}, "end": {"value": 10}},
            },
        ]
    }
    specs = _parse_act_site_residues(entry)
    assert [s["seq_pos"] for s in specs] == [212, 248]


def test_parse_act_site_legacy_token():
    entry = {
        "features": [
            {
                "type": "ACT_SITE",
                "location": {"start": {"value": 5}, "end": {"value": 5}},
            }
        ]
    }
    assert _parse_act_site_residues(entry)[0]["seq_pos"] == 5


def test_link_header_with_commas_in_fields():
    link = (
        '<https://rest.uniprot.org/uniprotkb/search?fields=accession,id,ec,'
        "sequence,ft_act_site,xref_pdb,protein_name&query=%28ft_act_site%3A%2A%29"
        '&cursor=abc&size=50>; rel="next"'
    )
    next_url = None
    for m in re.finditer(r'<([^>]+)>\s*;\s*rel="([^"]+)"', link):
        if m.group(2) == "next":
            next_url = m.group(1)
            break
    assert next_url is not None
    assert next_url.startswith("https://")
    assert "protein_name" in next_url
    assert "cursor=abc" in next_url
