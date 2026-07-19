ROOT := ..
ROOT_ABS := $(abspath $(ROOT))
PYTHON := $(ROOT_ABS)/.venv/bin/python

.PHONY: tables figures paper paper-full submission-paper submission-paper-full submission-preflight submission-finalize html clean

tables:
	cd $(ROOT_ABS) && $(PYTHON) paper/generate_tables.py --output-dir paper/generated

figures:
	cd $(ROOT_ABS) && $(PYTHON) -m finauth_audit.figures.plot_all --output-dir paper/figures

paper:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -jobname=FinAuth-Audit main_internal.tex

paper-full: tables figures paper

submission-paper:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -jobname=FinAuth-Audit main.tex

submission-paper-full: tables figures submission-paper

submission-preflight:
	$(PYTHON) finalize_submission.py --check-only

submission-finalize:
	$(PYTHON) finalize_submission.py

html:
	cd $(ROOT_ABS) && $(PYTHON) paper/build_html.py --output-dir paper/html

clean:
	latexmk -C -jobname=FinAuth-Audit main_internal.tex
	latexmk -C -jobname=FinAuth-Audit main.tex
	rm -f FinAuth-Audit.pdf
