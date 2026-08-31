# Directed-Evolution-loop

The goal of this repo is to modelize mathematically the directed evolution protocol used
at IDV for AAV, in order to optimize it.

## Repository structure

```
Directed-Evolution-loop/
├── Modelization_V1/              # active model — current work happens here
│   ├── pyproject.toml            # makes lib/ importable as `sequence_classesV1`, `analysisV1`, etc. from anywhere
│   ├── lib/                      # shared modules, imported bare (`from sequence_classesV1 import *`) by every notebook below
│   │   ├── sequence_classesV1.py      # core simulation engine: Protocol/ProtocolV2/ProtocolV3, initialize_random_weights
│   │   ├── analysisV1.py              # analysis/plotting helpers: pearson, precision_at_k, plot_teacher_weights, ...
│   │   ├── RegressionV1.py            # ridge-regression dataset builders
│   │   ├── initialize_weights.py      # loads real AAV9 F_viab/J_viab, builds correlated/anticorrelated/indep F_sel/J_sel
│   │   ├── MLP_regV1.py               # early/abandoned profile-only MLP attempt
│   │   └── aav{2,5,9}_{F,J}_viab_mlp.npy  # tracked in git (tiny) — derived artifacts, see "Data & derived artifacts" below
│   ├── notebooks/
│   │   ├── aav_viability_test/        # real AAV2/5/9 data: profile models (train ProfileMLP, export F_viab/J_viab),
│   │   │                              #   fitting-protocol calibration, Potts-regression GT + GT-score study, ...
│   │   ├── directed_evolution_loop/   # DE_loopV1.ipynb — the end-to-end directed-evolution simulation loop
│   │   ├── mlp_regression/            # MLP recovery experiments (DEEPMLP, ProfileMLP_recovery_nnx, ...)
│   │   │   └── claude_variants/       # AI-assisted exploratory variants of the same experiments
│   │   ├── reproducibility/           # NGS replicate reproducibility + noise-source analysis (real AAV9 data)
│   │   ├── selectivity_weight_regimes/# F_sel/J_sel correlated / anticorrelated / independent trio, + playground + plotting
│   │   └── viability_parameter_sweeps/# mu/T_viab/noise_viab/D/diversity sweeps + unified sweep + shallow-vs-deep MLP
│   ├── docs/                     # reference PDFs/tex (weight extraction, one-hot encoding, protocol diagram, ...)
│   └── contrib/                  # Yassine04_08_2026.py — a collaborator's standalone Colab export, not imported elsewhere
└── V0_prototype/                 # first-generation prototype, kept for history — imports are already broken
                                   # (`First_modelization` package no longer exists) and it isn't maintained
```

## Setup from a fresh clone

1. **Python 3.12** and a CUDA-capable GPU (JAX is pinned here with the `cuda13` extra —
   adjust that extra in `Modelization_V1/pyproject.toml` to match your machine's CUDA
   version if it differs; see JAX's own installation docs for the available extras).

2. Create a virtual environment and install the project (from the repo root):
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -e Modelization_V1/
   ```
   This installs every dependency (JAX/Flax/Optax, pandas, scikit-learn, Jupyter, ...)
   **and** makes `lib/`'s modules importable as bare names (`from sequence_classesV1
   import *`, `from analysisV1 import *`, ...) from any notebook under `notebooks/`,
   regardless of which subfolder it lives in.

3. *(Optional but recommended)* Register the venv as a Jupyter kernel, so notebooks in
   VS Code / JupyterLab can select it:
   ```bash
   python -m ipykernel install --user --name directed-evolution --display-name "Directed Evolution (.venv)"
   ```

## Data & derived artifacts

1. **Raw AAV NGS data** (`aav2.csv`, `aav5.csv`, `aav9.csv`, `fit4functionaav9.csv`) —
   gitignored, too large to track in git (`aav5.csv` alone is ~90MB). Published instead as
   assets on the
   [`aav-raw-ngs-data-v1` release](https://github.com/merlin-cc/Directed-Evolution-loop/releases/tag/aav-raw-ngs-data-v1).
   Download and place each file directly in its notebook folder (the first 3 in
   `Modelization_V1/notebooks/aav_viability_test/`, `fit4functionaav9.csv` in
   `Modelization_V1/notebooks/reproducibility/`):
   ```bash
   gh release download aav-raw-ngs-data-v1 --repo merlin-cc/Directed-Evolution-loop \
     --dir /tmp/aav-raw-ngs-data
   mv /tmp/aav-raw-ngs-data/aav{2,5,9}.csv Modelization_V1/notebooks/aav_viability_test/
   mv /tmp/aav-raw-ngs-data/fit4functionaav9.csv Modelization_V1/notebooks/reproducibility/
   ```
   (or download them by hand from the release page above, no `gh` required).
2. **`aav{2,5,9}_F_viab_mlp.npy` / `aav{2,5,9}_J_viab_mlp.npy`** (naive per-cell GT) and
   **`aav9_F_viab_potts.npy` / `aav9_J_viab_potts.npy`** (joint ridge/Potts-regression GT,
   the project's current default for AAV9 — better held-out predictive accuracy than the
   naive one, see `Modelization_V1/notebooks/aav_viability_test/AAV9_potts_regression.ipynb`)
   in `Modelization_V1/lib/` — tiny (a few hundred KB total) derived weight arrays, **tracked
   directly in git**, so a fresh clone has them without any extra step. Only regenerate them
   if you actually need to: the naive ones come from `AAV{2,5,9}_profile_model.ipynb`'s export
   cell (section 3c, right after `F_viab_GT`/`J_naive_final` are built) — it trains
   `ProfileMLP` on that AAV's raw CSV from step 1 as a scan target and overwrites the naive
   `.npy` pair; the Potts ones come from `AAV9_potts_regression.ipynb`'s export cell (a joint
   ridge fit via `RegressionV1.fit_weights_potts_from_data`, ~10 min). Both write into
   `Modelization_V1/lib/`, where `initialize_weights.py` expects them.
3. **`diversity*.csv` datasets** in `notebooks/selectivity_weight_regimes/` and
   `notebooks/viability_parameter_sweeps/` — gitignored, auto-generated (and cached) the
   first time each of `MLP_for_correlated_weights.ipynb` / `MLP_for_anticorrelated_weights.ipynb`
   / `MLP_for_independent_weights.ipynb` / `F_permutation_recovery_correlation.ipynb` runs.
   The first run of each is slow (MLP training + a brute-force scan over the full
   `20**7` sequence space, ~15–20 min); later runs just load the cached CSV. These caches are
   keyed by hyperparameters but not by which GT (`F_viab`/`J_viab`) produced them — if you
   swap the GT source again, delete the stale `diversity*.csv` files for any notebook you
   plan to re-run, otherwise it will silently reload the old GT's cached results.
