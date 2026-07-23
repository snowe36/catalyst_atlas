"""Console entry points: cat-download, cat-sites, cat-embed, cat-eval, cat-search, cat-figures, cat-design-*."""

from __future__ import annotations

import argparse
import logging
import sys


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def download_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize the high-confidence catalytic atlas"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="Build the high-confidence demo atlas (default; quality-first)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Ingest curated M-CSA sites + RCSB structures (falls back to demo on failure)",
    )
    parser.add_argument(
        "--n-enzymes",
        type=int,
        default=800,
        help="Demo size, or max M-CSA entries when using --public",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--expanded",
        action="store_true",
        help="Merge UniProt ACT_SITE extras + EC labels / structure_source (public mode)",
    )
    parser.add_argument(
        "--n-extra",
        type=int,
        default=200,
        help="Max UniProt-sourced extras when using --expanded",
    )
    parser.add_argument(
        "--no-alphafold",
        action="store_true",
        help="When expanding, skip AlphaFold fallback structures",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.data.download import download_atlas

    download_atlas(
        demo=not args.public,
        n_enzymes=args.n_enzymes,
        seed=args.seed,
        expanded=args.expanded,
        n_extra=args.n_extra,
        allow_alphafold=not args.no_alphafold,
    )
    return 0


def sites_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract catalytic microenvironments")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.site.extract import run_site_extraction

    run_site_extraction()
    return 0


def embed_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build engineered catalytic embeddings / retrieval index"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.models.embed import run_embed

    run_embed()
    return 0


def eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Leakage-aware chemistry identification evaluation"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--no-external",
        action="store_true",
        help="Skip MMseqs2/Foldseek retrieval baselines",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Label column (default: chemistry_family; try ec_class / ec3 after expand)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.eval.run import run_eval

    results = run_eval(
        k=args.k,
        test_size=args.test_size,
        seed=args.seed,
        run_external=not args.no_external,
        threads=args.threads,
        label_col=args.label_col,
    )
    # Print a compact summary for the terminal / README scraping.
    for split, payload in results["splits"].items():
        methods = payload["methods"]
        cat = methods["catalyst_microenvironment"]["accuracy"]
        mm = methods.get("mmseqs_transfer", {}).get("accuracy", float("nan"))
        fs = methods.get("foldseek_transfer", {}).get("accuracy", float("nan"))
        esm = methods.get("esm2_transfer", {}).get("accuracy", float("nan"))
        learned = methods.get("learned_catalytic_encoder", {}).get("accuracy", float("nan"))
        hybrid = methods.get("catalyst_hybrid", {}).get("accuracy", float("nan"))
        fusion = methods.get("learned_fusion_encoder", {}).get("accuracy", float("nan"))
        esm_gnn = methods.get("esm_gnn_fusion", {}).get("accuracy", float("nan"))
        print(
            f"{split:14s}  catalyst={cat:.3f}  esm_gnn={esm_gnn:.3f}  "
            f"esm={esm:.3f}  hybrid={hybrid:.3f}  fusion={fusion:.3f}  "
            f"learned={learned:.3f}  mmseqs={mm:.3f}  foldseek={fs:.3f}"
        )
    ann = results.get("annotation_style_audits") or {}
    if ann:
        print("annotation_style_audits (random):")
        for key, block in ann.items():
            n = block.get("n", 0)
            methods = block.get("methods") or {}
            bits = "  ".join(f"{m}={v['accuracy']:.3f}" for m, v in methods.items())
            print(f"  {key:40s} n={n:<4}  {bits}")
    return 0


def search_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Identify chemistry for a protein from its catalytic microenvironment"
    )
    parser.add_argument("--enzyme-id", type=str, default=None)
    parser.add_argument(
        "--demo-hero",
        action="store_true",
        help="Emit the cryptic-chemistry README hero case",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.search import run_search

    try:
        text = run_search(args.enzyme_id, args.demo_hero, args.k)
    except (KeyError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(text)
    return 0


def figures_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render catalytic microenvironment structure figures (offline, no PyMOL)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution (default: 180)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.viz.readme_figures import generate_readme_figures
    from catalyst_atlas.viz.retrieval_figure import generate_retrieval_figure
    from catalyst_atlas.viz.structure_figures import generate_structure_figures

    paths: list = []
    try:
        paths.extend(generate_structure_figures(dpi=args.dpi))
        paths.extend(generate_readme_figures(dpi=args.dpi))
        try:
            paths.append(generate_retrieval_figure(dpi=args.dpi))
        except (FileNotFoundError, RuntimeError) as exc:
            # Optional — needs ESM + ESM+GNN embeddings.
            print(f"retrieval figure skipped: {exc}", file=sys.stderr)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


def enrich_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enrich atlas with PDB cofactors/metals + chemistry ontology labels"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.data.enrich import enrich_atlas_cofactors_and_ontology

    df = enrich_atlas_cofactors_and_ontology()
    print(
        f"enriched n={len(df)}  "
        f"with_cofactors={(df['cofactor_tags'] != 'none').sum()}  "
        f"families={df['chemistry_family'].nunique()}"
    )
    return 0


def cases_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write three real M-CSA chemistry case studies"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.case_studies import write_case_studies

    try:
        path = write_case_studies(k=args.k)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


def graphs_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build explicit reaction-center graphs (catalytic machine graphs)"
    )
    parser.add_argument(
        "--max-first-shell",
        type=int,
        default=None,
        help="Cap nearest first-shell residues (default: graphs.MAX_FIRST_SHELL=4)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.featurize.graphs import run_graphs

    summary = run_graphs(max_first_shell=args.max_first_shell)
    print(
        f"graphs n={summary['n_enzymes']}  "
        f"mean_nodes={summary['mean_nodes']:.1f}  "
        f"mean_edges={summary['mean_edges']:.1f}  "
        f"node_dim={summary['node_dim']}  "
        f"max_first_shell={summary['max_first_shell']}"
    )
    return 0


def esm_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Frozen ESM-2 sequence embeddings (chemistry-transfer control)"
    )
    parser.add_argument("--model", type=str, default="esm2_t12_35M_UR50D")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.models.esm_embed import run_esm_embed

    try:
        summary = run_esm_embed(
            model_name=args.model, batch_size=args.batch_size, device=args.device
        )
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"esm n={summary['n_enzymes']} dim={summary['dim']} model={summary['model']}")
    return 0


def train_encoder_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train reaction-center graph encoder (supervised contrastive)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="fold_cluster",
        help="Leakage-aware split whose train set is used (default: fold_cluster)",
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--val-folds",
        type=int,
        default=12,
        help="Number of fold_clusters held out from train for validation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early-stop patience on validation chemistry accuracy",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=100,
        help="Do not early-stop before this many epochs",
    )
    parser.add_argument(
        "--lambda-cls",
        type=float,
        default=0.3,
        help="Weight on chemistry classification auxiliary loss",
    )
    parser.add_argument(
        "--fusion",
        action="store_true",
        help="Fuse engineered features_full into GNN readout; write embedding_fusion.npy",
    )
    parser.add_argument(
        "--fusion-side",
        action="store_true",
        help="Fuse metal/cofactor side vector only (less fold leakage than --fusion)",
    )
    parser.add_argument(
        "--fusion-esm",
        action="store_true",
        help="Fuse frozen ESM-2 embedding into GNN readout; write embedding_esm_gnn.npy",
    )
    parser.add_argument(
        "--random-graphs",
        action="store_true",
        help="Ablation with --fusion-esm: shuffle node features within each graph",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Run all epochs; select best among checkpoints every --checkpoint-every",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save a snapshot every N epochs for bake-off (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="SupCon temperature",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.models.train_encoder import train_reaction_center_encoder

    try:
        summary = train_reaction_center_encoder(
            split=args.split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            device=args.device,
            n_val_folds=args.val_folds,
            patience=args.patience,
            min_epochs=args.min_epochs,
            lambda_cls=args.lambda_cls,
            fusion=args.fusion,
            fusion_side=args.fusion_side,
            fusion_esm=args.fusion_esm,
            random_graphs=args.random_graphs,
            no_early_stop=args.no_early_stop,
            checkpoint_every=args.checkpoint_every,
            temperature=args.temperature,
        )
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"trained split={summary['split']} n={summary['n_enzymes']} "
        f"dim={summary['embed_dim']} fusion_esm={summary.get('fusion_esm')} "
        f"selected={summary.get('selected_checkpoint')} "
        f"best_val={summary.get('best_val_acc')}@{summary.get('best_epoch')} "
        f"loss={summary['final_loss']}"
    )
    return 0


def design_pockets_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build catalytic pocket artifacts (fixed catalytic + redesignable shells)"
    )
    parser.add_argument(
        "--enzyme-id",
        action="append",
        default=None,
        help="Limit to enzyme id(s); repeatable. Default: full atlas.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.design.pocket import run_pockets

    df = run_pockets(enzyme_ids=args.enzyme_id)
    print(f"pockets n={len(df)} mean_redesignable={df['n_redesignable'].mean():.1f}")
    return 0


def design_generate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or import shell-only designs (catalytic residues fixed)"
    )
    parser.add_argument(
        "--enzyme-id",
        action="append",
        default=None,
        help="Enzyme id(s); default: resolved design panel",
    )
    parser.add_argument("--n-sequences", type=int, default=100)
    parser.add_argument(
        "--from-sequences",
        type=str,
        default=None,
        help="Import designs from FASTA instead of running ProteinMPNN",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate valid mock shell designs (CI / offline)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--panel-size", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from pathlib import Path

    from catalyst_atlas.design.generate import run_generate
    from catalyst_atlas.design.panel import resolve_panel

    if args.enzyme_id:
        eids = args.enzyme_id
    else:
        eids = [p["enzyme_id"] for p in resolve_panel(target_size=args.panel_size)]

    try:
        df = run_generate(
            eids,
            n_sequences=args.n_sequences,
            from_sequences=Path(args.from_sequences) if args.from_sequences else None,
            use_mock=args.mock,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"designs n={len(df)} enzymes={df['enzyme_id'].nunique()}")
    return 0


def design_funnel_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Chemistry-constrained funnel: hard filters + ESM/chemistry rank → AF shortlist"
        )
    )
    parser.add_argument("--enzyme-id", action="append", default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Designs per enzyme to carry into AF (default: 10)",
    )
    parser.add_argument("--max-mutations", type=int, default=40)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.design.funnel import run_funnel

    try:
        meta = run_funnel(
            top_k=args.top_k,
            max_mutations=args.max_mutations,
            enzyme_ids=args.enzyme_id,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"funnel in={meta['n_input_designs']} hard_pass={meta['n_passed_hard_filter']} "
        f"af={meta['n_af_designs']}+{meta['n_af_wt']}wt → {meta['paths']['af_fasta']}"
    )
    return 0


def design_score_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score designs with chemistry_constraint_score (WT baseline first)"
    )
    parser.add_argument("--enzyme-id", action="append", default=None)
    parser.add_argument(
        "--mock-predictions",
        action="store_true",
        help="Write placeholder AF metrics when real predictions are absent",
    )
    parser.add_argument(
        "--af-queue-only",
        action="store_true",
        help="Score only the funnel AF shortlist (+ WT), not all generative designs",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.design.score import run_score

    try:
        df = run_score(
            args.enzyme_id,
            mock_predictions=args.mock_predictions,
            seed=args.seed,
            af_queue_only=args.af_queue_only,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    n_des = int((~df["is_wt"]).sum())
    print(
        f"scored rows={len(df)} designs={n_des} "
        f"mean_score={df.loc[~df['is_wt'], 'chemistry_constraint_score'].mean():.3f}"
    )
    return 0


def design_run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end redesign case study (generate → funnel → AF rank → report)"
    )
    parser.add_argument("--panel-size", type=int, default=10)
    parser.add_argument("--n-sequences", type=int, default=100)
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="AF shortlist size per enzyme after cheap ranking",
    )
    parser.add_argument("--max-mutations", type=int, default=12)
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock generator + AF metrics (default for offline)",
    )
    parser.add_argument(
        "--no-mock",
        action="store_true",
        help="Require real imported sequences/predictions",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.design.report import run_design_pipeline

    mock = not args.no_mock
    try:
        summary = run_design_pipeline(
            target_size=args.panel_size,
            n_sequences=args.n_sequences,
            mock=mock,
            seed=args.seed,
            top_k=args.top_k,
            max_mutations=args.max_mutations,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    funnel = summary.get("funnel") or {}
    print(
        f"design case study enzymes={len(summary['panel'])} "
        f"af_shortlist={summary['n_designs']} "
        f"funnel_in={funnel.get('n_input_designs')} "
        f"report={summary['report']}"
    )
    return 0
