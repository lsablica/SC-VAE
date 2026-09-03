PYTHON ?= python3

.PHONY: help test package verify-results website figures-latent figures-mnist figures-contamination train-mnist train-smallnorb

help:
	@echo "test package verify-results website"
	@echo "figures-latent figures-mnist figures-contamination"
	@echo "train-mnist train-smallnorb (explicit long-running jobs)"

test:
	$(PYTHON) -m pytest -q

verify-results:
	$(PYTHON) -m pytest -q tests/test_final_artifacts.py tests/test_paper_results_fixture.py

package:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

website:
	$(PYTHON) -m experiments.mnist.interactive.build_site

figures-latent:
	$(PYTHON) -m experiments.latent_layer.figures

figures-mnist:
	$(PYTHON) -m experiments.mnist.generate_qualitative_figures

figures-contamination:
	$(PYTHON) -m experiments.contaminated_directional.figures

train-mnist:
	$(PYTHON) -m experiments.mnist.train --preset benchmark_comparison $(ARGS)

train-smallnorb:
	$(PYTHON) -m experiments.smallnorb.run_all $(ARGS)
