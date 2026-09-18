# Modelization_V2

Clean, Potts-regression-only successor to `Modelization_V1`, created 2026-08-31. Self-contained:
nothing here imports from or reads paths under `Modelization_V1/` — see "How it stays
self-contained" below.

## Why this exists

`Modelization_V1` has two different histories tangled together:

- an **old system**: a naive per-cell ground truth (`F` first as a group-mean, `J` afterward
  as an independent residual, with ad-hoc shrinkage) plus, separately, a **double-mutant-scan**
  method (`extract_effective_F`/`extract_effective_FJ_mlp`) that recovers effective F/J weights
  from a trained `ProfileMLP` by numerically perturbing real background sequences one or two
  positions at a time and differencing predictions.
- a **new system**: a single joint ridge-regularized linear (Potts) regression
  (`RegressionV1.fit_weights_potts_from_data`) fit directly on real `(sequence, log enrichment)`
  data — see "The Potts regression, in detail" below.

`Modelization_V1` keeps both, for history. `Modelization_V2` keeps **only** the new system:
every notebook here builds or uses `F`/`J` exclusively via the joint Potts regression.
**No trace of the double-mutant-scan method is included.**

## What's included, and why

| File | Why it's here |
|---|---|
| `notebooks/AAV9_potts_regression.ipynb` | Builds the GT: joint ridge regression of `F`/`J` directly on real `aav9.csv` data. The source of `lib/aav9_{F,J}_viab_potts.npy`. |
| `notebooks/AAV9_potts_GT_score_study.ipynb` | Uses that GT (loaded, not re-fit) to study the deterministic GT score and one simulated `Protocol` run against it. |
| `notebooks/AAV9_potts_GT_fitting_protocol.ipynb` | Uses that GT to replay the manual-playground / population-size / cross-packaging checks against real `aav9.csv`. |
| `notebooks/…/AAV9_poisson_regression_GT_score_study.ipynb` | A **rejected alternative**: infers the same Potts model by a Poisson (and Gamma) GLM instead of Gaussian ridge. Runs the score-study comparison — Poisson `r ≈ 0.75` vs ridge `0.89`. See "Tried and rejected" below. Writes gitignored `lib/aav9_{F,J}_viab_{poisson,gamma}.npy`. |
| `notebooks/aav9.csv` | Raw real AAV9 NGS data all three notebooks read. Gitignored (`*.csv`), same as in `Modelization_V1` — present here locally, but on a fresh clone download it the same way (`aav-raw-ngs-data-v1` GitHub release, see the top-level `README.md`) and drop it in `Modelization_V2/notebooks/`. |
| `lib/RegressionV1.py`, `lib/analysisV1.py`, `lib/sequence_classesV1.py`, `lib/initialize_weights.py`, `lib/cross_packaging_draft.py` | Infrastructure the three notebooks import (Potts regression, plotting/analysis helpers, the `Protocol` simulation engine, npy loaders, and the cross-packaging noise-source variant used in section 8 of the fitting-protocol notebook). None of these files contain double-mutant-scan code. |
| `lib/aav9_F_viab_potts.npy` / `aav9_J_viab_potts.npy` | The Potts GT itself. |
| `lib/aav9_F_viab_mlp.npy` / `aav9_J_viab_mlp.npy` | The **old naive** GT — kept only because `AAV9_potts_regression.ipynb` compares against it internally (sections 3–5: predictive accuracy, credibility check) as part of *verifying* the new regression is actually better. This is a comparison baseline, not a use of the double-mutant-scan method. |

**Left out** (selection rule: any file that was ambiguous, or that pulls in mutant-scan
code, was excluded rather than guessed at):

- `AAV2_profile_model.ipynb` / `AAV5_profile_model.ipynb` and the `AAV9_profile_model.ipynb`
  they're modeled on — all three still contain `extract_effective_F`/`extract_effective_FJ_mlp`
  (double-mutant scan) in their MLP-recovery sections.
- `AAV2_fitting_protocol.ipynb` / `AAV5_fitting_protocol.ipynb` — still load the old
  `aav{2,5}_F_viab_mlp.npy`/`_J_viab_mlp.npy` weights, not a Potts GT.
- `aav2.csv` / `aav5.csv` and their `_potts.npy` weights — exist in `Modelization_V1/lib/`, but
  no notebook migrated here needs them (no aav2/aav5 notebook was clean of the mutant-scan
  method), so they weren't copied.
- `AAV9_cross_packaging_and_hallucination_impact.ipynb`, `AAV9_FJ_matrix_top500_check.ipynb`,
  the `viability_parameter_sweeps/`/`selectivity_weight_regimes/` notebook families, etc. — not
  copied; out of scope for this migration (GT-construction methodology), not audited one by one.
- `AAV9_fitting_protocol.ipynb` itself (the naive-GT original) — stays in V1 by design, it's
  the historical document of how `mu`/`T_viab`/`noise_viab`/`D` were calibrated.

## How it stays self-contained

Each notebook's setup cell does `sys.path.insert(0, os.path.abspath("../lib"))` — this is a
**functioning** path insertion (unlike `Modelization_V1`'s notebooks, where the equivalent line
is vestigial and the real import mechanism is `Modelization_V1`'s own editable pip install).
`Modelization_V2` intentionally has no editable pip install wired up in the shared virtualenv
(to avoid a module-name collision with `Modelization_V1`'s identically-named `RegressionV1`,
`analysisV1`, etc.) — `sys.path.insert` alone is what makes every `import RegressionV1` /
`from sequence_classesV1 import ProtocolV3` / etc. resolve to *this* folder's copies. Verified
empirically (see the migration report) by importing every module fresh and asserting
`__file__` points under `Modelization_V2/`, not `Modelization_V1/`.

A `pyproject.toml` is still included, matching `Modelization_V1`'s convention, in case you want
`pip install -e Modelization_V2` for IDE autocomplete/type-checking later — it's optional, not
required to run the notebooks.

## The Potts regression, in detail

`RegressionV1.fit_weights_potts_from_data(seq_matrix, target, seed=0)` fits a single-site
(additive, `F`) + pairwise (epistatic, `J`) decomposition of the real log-enrichment `target`
directly, in one joint ridge regression — not `F` first and `J` afterward as an independent
residual.

**Design matrix** (`build_potts_features`): for `L=7` positions and `A=20` amino acids, each
sequence is expanded into `L*A = 140` single-site one-hot indicators + `(L choose 2)*A*A =
8,400` pairwise-interaction one-hot indicators (one column per `(position i, position j,
amino acid a, amino acid b)` combination actually representable) + a bias term — 8,541
columns total. This is exactly the "Potts model" parameterization from statistical physics,
adapted to protein sequence covariation: `h_i(a)` fields and `J_ij(a,b)` couplings, originally
popularized for inferring residue-residue contacts from multiple-sequence alignments via
message-passing / pseudolikelihood maximization (Weigt et al., *PNAS* 2009,
[10.1073/pnas.0805923106](https://doi.org/10.1073/pnas.0805923106)). Here the same
parameterization is fit differently: by **linear ridge regression against a real scalar
fitness/enrichment readout** (`target`) rather than by maximizing a likelihood over aligned
homologous sequences — the same approach used to infer additive+pairwise fitness landscapes
directly from deep mutational scanning data (Otwinowski & Plotkin, *PNAS* 2014,
[10.1073/pnas.1400849111](https://doi.org/10.1073/pnas.1400849111), who also show and
quantify the bias regression-based epistasis estimates carry) and, most directly analogous to
what this repo does, to infer protein contacts from deep-mutagenesis fitness scores by
regularized regression of exactly this additive+pairwise model, then reading off the largest
`J` couplings (Rollins et al., *Nature Genetics* 2019,
[10.1038/s41588-019-0432-9](https://doi.org/10.1038/s41588-019-0432-9)).

**Fitting procedure** (`ridge_cv_mse_potts` + `fit_weights_potts`):
1. Build the design matrix `X` (8,541 columns) and target vector `y` = real `target`.
2. 5-fold cross-validated ridge regression over `lambdas_grid = np.logspace(-1, 2, 30)`
   (bias column never penalized) — picks the L2 strength minimizing held-out MSE.
3. Refit on the *entire* dataset at that best lambda: solve the regularized normal equations
   `(XᵀX + λI) β = Xᵀy` once, directly (`np.linalg.solve`), no iterative optimizer.
4. Unpack the flat coefficient vector `β` back into `F` (`L×A`) and `J` (`L×L×A×A`, symmetric,
   zero diagonal) tensors.

Ridge regularization matters structurally here, not just as a tuning preference: with
`aav9.csv`'s 68,776 real sequences against 8,541 features, the unregularized design is still
**rank-deficient** (rank 7,715/8,541, verified in `AAV9_potts_regression.ipynb` section 2) —
many `(i, j, a, b)` combinations are simply never co-observed in the real combinatorial
library, so an unpenalized fit would be non-unique. Ridge resolves this by construction: an
all-zero design column (a cell with zero support) contributes zero to `XᵀX`'s corresponding
row/column, and the `+λI` term alone determines that coefficient, driving it to exactly 0 —
no cell is guessed at from data that doesn't exist for it.

### Update 2026-09-18: matrix-free solver is now the default fitting method

The fitting procedure above (`build_potts_features` → dense `X` → `ridge_cv_mse_potts`/
`fit_weights_potts`/`fit_weights_potts_unregularized`, all via `RegressionV1.fit_weights_potts_from_data`)
still works and is unchanged — but it is no longer the recommended entry point for **new** Potts
regression fitting. Measured empirically (2026-09-18): materializing the dense design matrix `X`
(`N × 8,541`, float32) and solving via SVD (`lstsq`) or `np.linalg.solve` on `XᵀX` costs roughly
**5× `X`'s own size in peak RSS** (14.6 GiB at `N=90,000`, 40.1 GiB at `N=250,000` — a consistent
ratio, not just `X` itself sitting in memory), capping practical `N` at ~700-750k rows on a 121 GiB
machine, and requiring ~136 GiB for `X` alone at the scale of AAV2's organoid CSV (4.27M rows) —
not tractable at all.

`RegressionV1.fit_weights_potts_from_data_matrixfree` is a **drop-in replacement** — identical call
signature and return shape (`F_hat, J_hat, rank, info`) — that solves the exact same ridge objective
(`0.5·Σw(Xθ−y)² + 0.5·λ‖θ_no-bias‖²`) without ever materializing `X`: the forward pass is the same
gather-based computation every notebook in this project already hand-rolls as a local `score_FJ`
helper (now centralized as `RegressionV1.score_potts`), and the gradient (`jax.grad`, reverse-mode
autodiff through that gather) *is* the matrix-free `XᵀX`/`Xᵀy` operation — the same principle as the
"surrogate Potts" fit in `AAV9_cross_packaging_parameter_sweeps.ipynb` (which computes `Xᵀy` via
`np.bincount` instead), just obtained here via autodiff rather than hand-written bincounts. Solved
by L-BFGS-B (`scipy.optimize.minimize`) — a convex quadratic, so it converges to the *same* optimum
as the direct solve, just iteratively. K-fold CV (`lam=None`) is matrix-free end to end too: every
fold's fit *and* its validation-MSE scoring go through the same gather-based path, unlike
`ridge_cv_mse_potts`, which still builds a dense `X` per fold.

**Validated, not just asserted**: `AAVs dataset/AAV2/viability/AAV2_potts_ridge_matrixfree_validation.ipynb`
fits the same AAV2 data both ways (`N=90,000`, the old method's practical ceiling) and compares —
`r(F_classic, F_matrixfree)` / `r(J_classic, J_matrixfree)` **> 0.999999** in both the regularized
and unregularized (rank-deficient, minimum-norm) cases, held-out `r` identical to the 4th decimal.
The matrix-free solver was also **24-65× faster** at that same `N` (1.4-1.5s vs 36.5-91.0s) — a
side benefit, not the point. At the full AAV2 organoid scale (`N=4,153,463`), it fits in ~26s and
reaches `r=+0.302` held-out — the best AAV2 viability result obtained in this project to date,
beating both the memory-capped classic fit (`r=+0.202` at `N=90,000`) and a from-scratch
multinomial-MLE alternative explored the same day (`AAV2_potts_mle_multinomial.ipynb`, based on
Fernandez-de-Cossio-Diaz, Uguzzoni & Pagnani, *MBE* 2021,
[10.1093/molbev/msaa204](https://doi.org/10.1093/molbev/msaa204) — see that notebook's own
write-up for why it doesn't beat a properly-scaled ridge fit here, despite being the more
"principled" likelihood for count data on paper).

Consequently, every AAV2 notebook whose job is to *fit* F/J via Potts regression (as opposed to
consuming already-exported `.npy` weights downstream) was rebuilt to call
`fit_weights_potts_from_data_matrixfree` instead of `fit_weights_potts_from_data` — mechanically,
since the two share a signature. The pre-rebuild versions (still using the dense/SVD path) are kept
for reference in `AAVs dataset/AAV2/obsolete_dense_matrix_potts_regression/`. See the top-level
`CLAUDE.md`'s "État actuel" (2026-09-18 entries) for the full list and per-notebook detail. AAV5/AAV9
notebooks are unaffected — this project convention doesn't retroactively migrate work outside AAV2
unless/until asked.

### Tried and rejected: a Poisson-GLM fit

Ridge least squares on `log(target)` **is** a maximum-likelihood fit — of a linear model with
Gaussian noise on the log enrichment (`Modelization_V2/docs/GT_MLE_vs_pseudolikelihood.pdf`,
gitignored, spells this out, and also why pseudo-likelihood / plmDCA does not apply here —
that models `P(sequence)` over a sequence ensemble and ignores the measured label). The obvious alternative
likelihood is a **Poisson**: model the fold-enrichment *ratio* `exp(target)` as
`ratio ~ Poisson(exp(F·s + J·s·s + bias))`, i.e. a Tweedie GLM with a log link (`power=1`),
fit by `RegressionV1.fit_weights_glm_from_data` and studied in
`notebooks/notebooks/Viability/AAV9_poisson_regression_GT_score_study.ipynb` (the Gamma GLM,
`power=2`, is fit alongside it as a second reference point).

**It works clearly worse than the Gaussian ridge**, and on `aav9.csv` it is not really
salvageable:

| GT | `r`(deterministic GT score, real `target`) |
|---|---|
| ridge Potts (Gaussian-on-log, the default GT) | **0.889** |
| Poisson GLM | 0.753 |
| Gamma GLM | 0.759 |

`F` still comes out close to the ridge GT (`r ≈ 0.92–0.95`), but `J` collapses (`r(J, J_potts)
≈ 0.52`, couplings ~2× weaker). Cause: the Poisson deviance is scale-equivariant — with no
real per-sequence read counts in `aav9.csv` to set an offset, the loss is entirely determined
by the shape of the ratio distribution and is dominated by the handful of most-enriched
sequences, so the bulk of the library barely constrains the fit. (One real upside: precisely
*because* it is tail-weighted, the Poisson GT recovers the extreme top 1–10 % of the real
`target` ranking better than the ridge GT does — but it loses badly on everything else.) A
genuinely count-aware Poisson / Negative-Binomial fit would need raw plasmid/vector counts;
`aav2.csv`/`aav5.csv` expose those, `aav9.csv` does not. The experimental weights
(`lib/aav9_{F,J}_viab_{poisson,gamma}.npy`) are gitignored and no loader is wired up — the
ridge Potts GT stays the project GT.

## Regenerating

`AAV9_potts_regression.ipynb` writes the project GT — re-run it to refresh
`lib/aav9_F_viab_potts.npy`/`aav9_J_viab_potts.npy` after any change to `RegressionV1.py` or
to `aav9.csv`. `AAV9_poisson_regression_GT_score_study.ipynb` also writes `.npy`
(`aav9_{F,J}_viab_{poisson,gamma}.npy`), but those are gitignored experimental weights, not a
GT anything depends on.
