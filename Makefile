.PHONY: help install test figures paper clean lint

help:
	@echo "LARP: Latent Astrophysical Residual Pipeline"
	@echo "Available commands:"
	@echo "  make install     Install package in editable mode"
	@echo "  make test        Run fast unit test suite"
	@echo "  make figures     Regenerate all 8 publication paper figures"
	@echo "  make paper       Compile research paper markdown to docx via pandoc"
	@echo "  make audit       Audit production submission.csv schema"
	@echo "  make clean       Remove cached Python files and temporary data"

install:
	pip install -e ".[dev]"

test:
	python3 tests/run_tests.py

figures:
	python3 paper/scripts/generate_larp_paper_figures.py

paper:
	pandoc paper/LARP_Latent_Astrophysical_Residual_Pipeline.md \
		-o paper/LARP_Latent_Astrophysical_Residual_Pipeline.docx \
		--resource-path=paper

audit:
	python3 src/larp/cli.py audit --submission submission.csv

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
