# Archive

Superseded code, kept for reference. Do not cite results from these files.

- `simulation_v1.m`, `simulation_v2.m`, `simulation_v3.m` and `results_v*.png`
  are the first simulations of the GLM-with-aperiodic-covariates idea.
  `simulation_v3.m` scores recovery with the intercept of a regression whose
  predictors are all z-scored. That intercept is always the mean of the
  outcome, so it cannot distinguish between aperiodic estimators.
  `verify_v3_artifact.m` demonstrates this.
- `test_lambda_diag.m` was an early diagnostic of whether the coupling
  exponent can be identified. `code/test_lambda_identify.m` and
  `code/sim01_estimator_benchmark.m` replace it.

The current analyses are in `code/`; see the main README.
