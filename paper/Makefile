ROOT_ABS := $(abspath ..)
PYTHON := $(ROOT_ABS)/.venv/bin/python

SOURCE_DATE_EPOCH ?= 1784456716
FORCE_SOURCE_DATE ?= 1
export SOURCE_DATE_EPOCH
export FORCE_SOURCE_DATE

.PHONY: paper submission-paper submission-preflight submission-finalize html clean

paper: submission-paper

submission-paper:
	latexmk -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error -jobname=FinAuth-Audit main.tex

submission-preflight:
	$(PYTHON) finalize_submission.py --check-only

submission-finalize:
	$(PYTHON) finalize_submission.py

html:
	cd $(ROOT_ABS) && $(PYTHON) paper/build_html.py --output-dir paper/html

clean:
	latexmk -C -jobname=FinAuth-Audit main.tex
