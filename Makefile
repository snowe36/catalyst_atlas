.PHONY: install lint test reproduce clean

install:
	python -m pip install -U pip
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	MPLBACKEND=Agg pytest -q

reproduce:
	bash scripts/reproduce.sh

clean:
	rm -rf data/raw/* data/processed/*
	rm -rf out/*.md out/*.json out/*.log out/case_studies
	touch data/processed/.gitkeep
