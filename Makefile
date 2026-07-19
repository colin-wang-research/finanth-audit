PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,../.venv/bin/python)
PYTHONPATH := ..
SMOKE_CONFIG := configs/smoke.yaml

.PHONY: smoke provenance-smoke public-fetch public-audit training-smoke training-robustness main-validation robustness-validation generator-robustness paper-test-partition paper-test-rehearsal paper-test-freeze paper-test-once external-orderbook-fetch external-orderbook-build external-orderbook-audit external-orderbook-power external-orderbook-freeze external-orderbook-test real-agent-fetch real-agent-build real-agent-generate real-agent-audit real-agent-freeze real-agent-test verify-real-agent verify-real-agent-v06 verify-external-orderbook verify-main verify-paper verify-paper-internal verify-release submission-preflight submission-finalize data core full verify paper html release public-check clean-smoke

smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.controlled --config $(SMOKE_CONFIG)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.coverage_audit --config $(SMOKE_CONFIG)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.shortcut_audit --config $(SMOKE_CONFIG)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests

data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.controlled --config $(SMOKE_CONFIG)

provenance-smoke: data
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.provenance --config configs/provenance_smoke.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.provenance_audit --config configs/provenance_smoke.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests

public-fetch:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.fetch_polymarket_point_in_time --config configs/public_audit.yaml

public-audit: provenance-smoke public-fetch
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.public_polymarket --config configs/public_audit.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.point_in_time_audit --config configs/public_audit.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.public_replay --config configs/public_audit.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.public_power_gate --config configs/public_audit.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests

training-smoke: public-audit
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.training_corpora --config configs/training_utility_smoke.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.training_utility --config configs/training_utility_smoke.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests

training-robustness:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.training_robustness --config configs/training_robustness_v03.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests

main-validation:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.controlled --config configs/main.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.coverage_audit --config configs/main.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.shortcut_audit --config configs/main.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.provenance --config configs/provenance_main.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.provenance_audit --config configs/provenance_main.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.main_validation_summary
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests

robustness-validation:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.certification_robustness --config configs/main.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.review_workload
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.baseline_governance --config configs/main.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests

generator-robustness:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.mechanistic_families --config configs/generator_robustness.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.generator_robustness --config configs/generator_robustness.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.provenance_identifiability
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests

paper-test-partition:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.paper_test_partition

paper-test-rehearsal:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.paper_test --validation-rehearsal --bootstrap-replicates 200

paper-test-freeze:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.paper_test_freeze

paper-test-once:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.paper_test --execute-frozen-paper-test --freeze-manifest manifests/paper_test_freeze.json

external-orderbook-fetch:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.fetch_binance_depth_v03 --config configs/external_orderbook_v03.yaml

external-orderbook-build:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.build_binance_depth_v03 --config configs/external_orderbook_v03.yaml
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.build_databento_bbo_v03 --config configs/external_orderbook_v03.yaml

external-orderbook-audit:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.external_orderbook_structural_audit --config configs/external_orderbook_v03.yaml

external-orderbook-power:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.external_orderbook_power --config configs/external_orderbook_v03.yaml

external-orderbook-freeze:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.external_orderbook_freeze --config configs/external_orderbook_v03.yaml

external-orderbook-test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.external_orderbook_test --config configs/external_orderbook_v03.yaml --execute-frozen-test --freeze-manifest manifests/external_orderbook_v03_freeze.json

real-agent-fetch:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.fetch_binance_depth_v03 --config configs/real_agent_v05.yaml

real-agent-build:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.build_real_agent_contexts_v05 --config configs/real_agent_v05.yaml

real-agent-generate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.generators.generate_real_agent_proposals_v05 --config configs/real_agent_v05.yaml

real-agent-audit:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.real_agent_structural_audit_v05 --config configs/real_agent_v05.yaml

real-agent-freeze:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.real_agent_freeze_v05 --config configs/real_agent_v05.yaml

real-agent-test:
	@echo "One-time target: this command must fail after the completed registry exists."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.real_agent_test_v05 --config configs/real_agent_v05.yaml --execute-frozen-test --freeze-manifest manifests/real_agent_v05_freeze.json

verify-external-orderbook:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) verify_artifact.py --phase external-orderbook --run-tests

verify-real-agent:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) verify_artifact.py --phase real-agent --run-tests

verify-real-agent-v06:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) verify_artifact.py --phase real-agent-v06

verify-main:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.overlap_audit
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) verify_artifact.py --phase main --run-tests

core: smoke

full:
	@echo "Full v0.2.0 run remains gated until Phases 2-4 and the test registry are frozen."
	@exit 2

verify:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.evaluation.overlap_audit
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) verify_artifact.py --phase public --run-tests

paper:
	$(MAKE) -C paper paper

verify-paper: paper html
	$(PYTHON) paper/verify_paper.py

verify-paper-internal: submission-preflight
	$(MAKE) -C paper submission-paper
	$(MAKE) -C paper html
	$(PYTHON) paper/verify_paper.py --allow-placeholder-authors

submission-preflight:
	$(PYTHON) paper/finalize_submission.py --check-only

submission-finalize:
	$(PYTHON) paper/finalize_submission.py

html: paper
	$(MAKE) -C paper html

public-check:
	$(PYTHON) -m compileall -q .
	$(PYTHON) -m pytest -q release_tests

release: public-check
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.release.build_release --output-dir dist
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.release.verify_release --archive dist/finauth-audit-$$(cat VERSION).tar.gz --report dist/finauth-audit-$$(cat VERSION).verification.json

verify-release:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m finauth_audit.release.verify_release --archive dist/finauth-audit-$$(cat VERSION).tar.gz --report dist/finauth-audit-$$(cat VERSION).verification.json

clean-smoke:
	@echo "Smoke outputs are retained as audit evidence; remove only through a versioned cleanup protocol."
