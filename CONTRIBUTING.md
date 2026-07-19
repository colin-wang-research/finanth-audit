# Contributing

FinAuth-Audit is a frozen benchmark and release protocol. Contributions are
welcome when they preserve the distinction between benchmark evaluation and
trading-strategy development.

## Before opening a pull request

1. Do not add credentials, raw provider responses, licensed row-level data,
   community-hidden material, or paper-test reruns.
2. Keep structural `N/A` values as `N/A`; do not convert them to zero.
3. Document any new rule's legal decision-time fields and source-role policy.
4. Add focused tests for code changes.
5. Run:

```bash
python -m pip install ".[test]"
python -m pytest -q release_tests
python -m compileall -q .
```

Scientific extensions should use a new versioned protocol and must not modify
the completed one-time paper-test outputs.
