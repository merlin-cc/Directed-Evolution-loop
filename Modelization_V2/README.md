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

## Regenerating

`AAV9_potts_regression.ipynb` is the only notebook that writes `.npy` files — re-run it (not
the other two) to refresh `lib/aav9_F_viab_potts.npy`/`aav9_J_viab_potts.npy` after any change
to `RegressionV1.py` or to `aav9.csv`.
