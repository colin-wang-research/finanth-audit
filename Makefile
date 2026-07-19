PYTHON ?= .venv/bin/python
PYTHONPATH := ..

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
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.release.build_release --output-dir dist
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.release.verify_release --archive dist/finauth-audit-$$(cat VERSION).tar.gz --report dist/finauth-audit-$$(cat VERSION).verification.json

verify-release:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.release.verify_release --archive dist/finauth-audit-$$(cat VERSION).tar.gz --report dist/finauth-audit-$$(cat VERSION).verification.json
