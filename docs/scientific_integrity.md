# Scientific Integrity

## Hypotheses are falsifiable

Expected outcomes must not be encoded into evaluation logic. A hypothesis remains conditional until an approved experiment measures it. Evaluation must be capable of showing no gain or harm.

## Baselines must remain honest

The reproduced baseline cannot be weakened, altered without disclosure, or evaluated under worse conditions to favor a proposed method. Research modifications remain outside baseline code and are compared against a faithful, reproducible implementation.

## Same-data fairness

Matched comparisons use the same dataset split, preprocessing, resolution, evaluation code, and hardware when compute is compared. Any unavoidable difference is documented and limits the conclusion.

## No test-set optimization

Test data must not determine calibration temperature, thresholds, hyperparameters, model selection, or stopping decisions. Validation data owns these choices. Dataset and sequence splits are explicit and must not be changed silently.

## Oracle is diagnostic or supervisory

Oracle interventions use measured counterfactual outcomes to estimate headroom, diagnose error sources, or supervise later decisions. Oracle performance is an upper bound or diagnostic result, not deployable performance. Oracle-only information must not leak into inference.

## Gates and negative results

Gate failures are preserved and reported. If stereo-derived covariance lacks headroom, repair utilities are not heterogeneous, or routing overhead erases savings, the corresponding claim is reduced or the approved pivot is followed. Architecture and metrics must not be distorted to rescue a hypothesis.

## Evidence and claims

Keep these states separate: code exists, an experiment ran, a result was verified, and a claim is supported. Metrics require retained provenance. Efficiency requires measured end-to-end runtime and memory evidence on recorded hardware; novelty claims require comparison with the approved literature position.

## Clinical scope

Development reconstruction, calibration, rendering, and runtime metrics do not establish clinical benefit or safety. Do not translate benchmark gains into unsupported claims about intraoperative performance, patient outcomes, diagnosis, or deployment readiness.
