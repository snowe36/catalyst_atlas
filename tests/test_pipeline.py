"""End-to-end tests on a small high-confidence demo atlas."""

from __future__ import annotations

from catalyst_atlas.data.download import download_atlas
from catalyst_atlas.eval.run import run_eval
from catalyst_atlas.models.embed import load_index, run_embed, transfer_chemistry
from catalyst_atlas.search import find_cryptic_hero, search_enzyme
from catalyst_atlas.site.extract import run_site_extraction


def test_demo_pipeline_chemistry_beats_seq_proxy_on_seq_holdout(tmp_path, monkeypatch):
    # Isolate writes under package paths by running against real PROCESSED —
    # pipeline is idempotent and CI-safe at n=240.
    download_atlas(demo=True, n_enzymes=240, seed=7)
    run_site_extraction()
    run_embed()
    results = run_eval(k=5, test_size=0.25, seed=7)

    seq_split = results["splits"]["seq_cluster"]
    cat_acc = seq_split["methods"]["catalyst_microenvironment"]["accuracy"]
    seq_acc = seq_split["methods"]["sequence_cluster_transfer"]["accuracy"]
    assert cat_acc > 0.5
    # Microenvironment should beat pure sequence-cluster transfer on the hard split.
    assert cat_acc >= seq_acc


def test_transfer_chemistry_card():
    download_atlas(demo=True, n_enzymes=120, seed=3)
    run_site_extraction()
    run_embed()
    index = load_index(composition_only=False)
    card = transfer_chemistry(index, 0, k=5)
    assert "predicted_chemistry_class" in card
    assert len(card["neighbors"]) == 5
    assert card["query_enzyme_id"]


def test_search_and_hero():
    download_atlas(demo=True, n_enzymes=180, seed=11)
    run_site_extraction()
    run_embed()
    index = load_index(composition_only=False)
    eid = index.meta.iloc[0]["enzyme_id"]
    card = search_enzyme(eid, k=3)
    assert card["query_enzyme_id"] == eid
    hero = find_cryptic_hero(k=5)
    assert hero["card"]["predicted_chemistry_class"]
