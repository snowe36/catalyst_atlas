"""Console entry points: cat-download, cat-sites, cat-embed, cat-eval, cat-search, cat-figures."""

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
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from catalyst_atlas.data.download import download_atlas

    download_atlas(demo=not args.public, n_enzymes=args.n_enzymes, seed=args.seed)
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
    )
    # Print a compact summary for the terminal / README scraping.
    for split, payload in results["splits"].items():
        methods = payload["methods"]
        cat = methods["catalyst_microenvironment"]["accuracy"]
        mm = methods.get("mmseqs_transfer", {}).get("accuracy", float("nan"))
        fs = methods.get("foldseek_transfer", {}).get("accuracy", float("nan"))
        print(
            f"{split:14s}  catalyst={cat:.3f}  "
            f"mmseqs={mm:.3f}  foldseek={fs:.3f}"
        )
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

    from catalyst_atlas.search import search_main_logic

    try:
        text = search_main_logic(args.enzyme_id, args.demo_hero, args.k)
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
    from catalyst_atlas.viz.structure_figures import generate_structure_figures

    paths: list = []
    try:
        paths.extend(generate_structure_figures(dpi=args.dpi))
        paths.extend(generate_readme_figures(dpi=args.dpi))
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
