PYTHON ?= .venv/bin/python
.PHONY: public-check paper html verify-paper submission-preflight submission-finalize release verify-release

public-check:
	$(PYTHON) -m compileall -q .
	$(PYTHON) -m pytest -q release_tests

paper:
	$(MAKE) -C paper submission-paper

html:
	$(MAKE) -C paper html

verify-paper: paper html
	$(PYTHON) paper/verify_paper.py

submission-preflight:
	$(PYTHON) paper/finalize_submission.py --check-only

submission-finalize:
	$(PYTHON) paper/finalize_submission.py

release: public-check
	$(PYTHON) release/build_release.py --output-dir dist
	$(PYTHON) release/verify_release.py --archive dist/finauth-audit-$$(cat VERSION).tar.gz --report dist/finauth-audit-$$(cat VERSION).verification.json

verify-release:
	$(PYTHON) release/verify_release.py --archive dist/finauth-audit-$$(cat VERSION).tar.gz --report dist/finauth-audit-$$(cat VERSION).verification.json
