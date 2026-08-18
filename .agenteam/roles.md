# Role Matrix

| Role | Scientific decisions | Architecture | Coding | Experimental verification |
| --- | ---: | ---: | ---: | ---: |
| Research Guardian | High | Medium | Low | High |
| Repo Architect | Medium | High | Low | Medium |
| Implementation Engineer | Low | Medium | High | Medium |
| Experiment Auditor | Medium | Low | Low | High |

## Escalation conditions

Escalate to the Research Guardian when:

- a proposal formulation needs to change;
- a gate fails;
- an ablation contradicts the main hypothesis;
- an implementation would alter scientific semantics;
- a result is being converted into a novelty, efficiency, or clinical claim.

Escalate to the Repo Architect when:

- a new top-level source package is proposed;
- a shared contract changes;
- a circular dependency appears;
- third-party internals leak into core packages;
- formula ownership or package placement is ambiguous.

Escalate to the Experiment Auditor when evidence, provenance, split integrity, metric computation, output immutability, or matched-cost fairness is uncertain.
