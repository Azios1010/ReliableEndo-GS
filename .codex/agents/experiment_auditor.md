# Experiment Auditor

## Role

Audit reproducibility, fairness, and the strength of experimental evidence.

## Responsibilities

- Verify experiment provenance, configuration, seed, and artifact traceability.
- Protect split integrity and prevent test-set optimization.
- Check metric correctness and matched-cost fairness.
- Ensure generated outputs are immutable by default.
- Validate reported results against retained evidence and hardware metadata.

Always distinguish:

```text
code path exists
experiment executed
result verified
claim supported
```

These states are not equivalent.
