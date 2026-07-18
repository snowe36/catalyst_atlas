"""End-to-end tests on a small high-confidence demo atlas."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from catalyst_atlas.data.download import download_atlas
from catalyst_atlas.data.generate_demo import _stable_int, generate_demo_atlas
from catalyst_atlas.eval.run import _scale_train_test, run_eval
from catalyst_atlas.models.embed import load_index, run_embed, transfer_chemistry
from catalyst_atlas.search import find_cryptic_hero, search_enzyme
from catalyst_atlas.site.extract import run_site_extraction


def test_stable_int_is_process_stable():
    a = _stable_int("hydrolase", salt=7)
    b = _stable_int("hydrolase", salt=7)
    assert a == b
    assert _stable_int("hydrolase", salt=7) != _stable_int("lyase", salt=7)
    assert _stable_int("x", mod=10) < 10


def test_demo_atlas_reproducible_across_calls():
    a = generate_demo_atlas(n_enzymes=40, seed=7)
    b = generate_demo_atlas(n_enzymes=40, seed=7)
    assert a["enzyme_id"].tolist() == b["enzyme_id"].tolist()
    assert a["seq_cluster"].tolist() == b["seq_cluster"].tolist()
    assert a["fold_cluster"].tolist() == b["fold_cluster"].tolist()
    # is_cryptic_seed tracks the actual cryptic assignment, not a second coin flip.
    assert a["is_cryptic_seed"].dtype == bool


def test_scale_train_test_fits_on_train_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    train_idx = np.arange(12)
    test_idx = np.arange(12, 20)
    X_train, X_test = _scale_train_test(X, train_idx, test_idx)
    ref = StandardScaler().fit(X[train_idx])
    np.testing.assert_allclose(X_train, ref.transform(X[train_idx]))
    np.testing.assert_allclose(X_test, ref.transform(X[test_idx]))
    # Must differ from a scaler fit on the full matrix (leakage check).
    leaked = StandardScaler().fit_transform(X)
    assert not np.allclose(X_test, leaked[test_idx])


def test_demo_pipeline_chemistry_beats_seq_proxy_on_seq_holdout(isolated_data_dirs):
    download_atlas(demo=True, n_enzymes=240, seed=7)
    run_site_extraction()
    run_embed()
    results = run_eval(k=5, test_size=0.25, seed=7)

    assert results["scaler"] == "StandardScaler fit on train split only"
    seq_split = results["splits"]["seq_cluster"]
    cat_acc = seq_split["methods"]["catalyst_microenvironment"]["accuracy"]
    seq_acc = seq_split["methods"]["sequence_cluster_transfer"]["accuracy"]
    assert cat_acc > 0.5
    # Microenvironment should beat pure sequence cluster-lookup on the hard split.
    assert cat_acc >= seq_acc


def test_transfer_chemistry_card(isolated_data_dirs):
    download_atlas(demo=True, n_enzymes=120, seed=3)
    run_site_extraction()
    run_embed()
    index = load_index(composition_only=False)
    card = transfer_chemistry(index, 0, k=5)
    assert "predicted_chemistry_class" in card
    assert len(card["neighbors"]) == 5
    assert card["query_enzyme_id"]


def test_search_and_hero(isolated_data_dirs):
    download_atlas(demo=True, n_enzymes=180, seed=11)
    run_site_extraction()
    run_embed()
    index = load_index(composition_only=False)
    eid = index.meta.iloc[0]["enzyme_id"]
    card = search_enzyme(eid, k=3)
    assert card["query_enzyme_id"] == eid
    hero = find_cryptic_hero(k=5)
    assert hero["card"]["predicted_chemistry_class"]
