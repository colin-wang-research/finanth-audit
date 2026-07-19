PACKAGE_ROOT_ABS := $(abspath ..)
ifneq ($(wildcard $(PACKAGE_ROOT_ABS)/.venv/bin/python),)
ROOT_ABS := $(PACKAGE_ROOT_ABS)
PYTHON := $(ROOT_ABS)/.venv/bin/python
PAPER_PREFIX :=
else
ROOT_ABS := $(abspath ../..)
PYTHON := $(ROOT_ABS)/.venv/bin/python
PAPER_PREFIX := finauth_audit/
endif

# Freeze PDF metadata and trailer IDs for byte-reproducible v0.6.0 builds.
SOURCE_DATE_EPOCH ?= 1784456716
FORCE_SOURCE_DATE ?= 1
export SOURCE_DATE_EPOCH
export FORCE_SOURCE_DATE

.PHONY: tables figures paper paper-full submission-paper submission-paper-full submission-preflight submission-finalize html clean

tables:
	cd $(ROOT_ABS) && $(PYTHON) $(PAPER_PREFIX)paper/generate_tables.py --output-dir $(PAPER_PREFIX)paper/generated

figures:
	cd $(ROOT_ABS) && $(PYTHON) -m finauth_audit.figures.plot_all --output-dir $(PAPER_PREFIX)paper/figures

paper:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -jobname=FinAuth-Audit main_internal.tex

paper-full: tables figures paper

submission-paper:
	latexmk -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error -jobname=FinAuth-Audit main.tex

submission-paper-full: tables figures submission-paper

submission-preflight:
	$(PYTHON) finalize_submission.py --check-only

submission-finalize:
	$(PYTHON) finalize_submission.py

html:
	cd $(ROOT_ABS) && $(PYTHON) $(PAPER_PREFIX)paper/build_html.py --output-dir $(PAPER_PREFIX)paper/html

clean:
	latexmk -C -jobname=FinAuth-Audit main_internal.tex
	latexmk -C -jobname=FinAuth-Audit main.tex
	rm -f FinAuth-Audit.pdf
