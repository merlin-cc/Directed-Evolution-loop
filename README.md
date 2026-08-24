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
│   │   ├── directed_evolution_loop/   # DE_loopV1.ipynb — the end-to-end directed-evolution simulation loop
│   │   ├── selectivity_weight_regimes/# F_sel/J_sel correlated / anticorrelated / independent trio, + playground + plotting
│   │   ├── aav_viability_test/        # AAV9_profile_model.ipynb — trains ProfileMLP on real AAV9 data, exports F_viab/J_viab
│   │   └── mlp_regression/            # MLP recovery experiments (DEEPMLP, ProfileMLP_recovery_nnx, ...)
│   │       └── claude_variants/       # AI-assisted exploratory variants of the same experiments
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

1. **Raw AAV NGS data** (`aav2.csv`, `aav5.csv`, `aav9.csv`) — gitignored, too large to
   track in git (`aav5.csv` alone is ~90MB). Published instead as assets on the
   [`aav-raw-ngs-data-v1` release](https://github.com/merlin-cc/Directed-Evolution-loop/releases/tag/aav-raw-ngs-data-v1).
   Download the 3 files and place them directly in
   `Modelization_V1/notebooks/aav_viability_test/`:
   ```bash
   gh release download aav-raw-ngs-data-v1 --repo merlin-cc/Directed-Evolution-loop \
     --dir Modelization_V1/notebooks/aav_viability_test/
   ```
   (or download them by hand from the release page above, no `gh` required).
2. **`aav{2,5,9}_F_viab_mlp.npy` / `aav{2,5,9}_J_viab_mlp.npy`** in `Modelization_V1/lib/` —
   tiny (a few hundred KB total) derived weight arrays, **tracked directly in git**, so a
   fresh clone has them without any extra step. Only regenerate them if you actually need
   to: run the corresponding `AAV{2,5,9}_profile_model.ipynb`'s export cell (section 3b,
   right after `F_mlp_fj, J_mlp = extract_effective_FJ_mlp(...)`) — it trains `ProfileMLP`
   on that AAV's raw CSV from step 1 and overwrites the `.npy` pair in `Modelization_V1/lib/`,
   where `initialize_weights.py` expects them.
3. **`diversity*.csv` datasets** in `notebooks/selectivity_weight_regimes/` — gitignored,
   auto-generated (and cached) the first time each of `MLP_for_correlated_weights.ipynb`
   / `MLP_for_anticorrelated_weights.ipynb` / `MLP_for_independent_weights.ipynb` runs.
   The first run of each is slow (MLP training + a brute-force scan over the full
   `20**7` sequence space, ~15–20 min); later runs just load the cached CSV.

## Next tasks

- modelize the pair-interactions between acido-amines. Pott's model ✅
- model (Ridge regression) to fit the scores for each sequence (first for the linear model) ✅
- Poisson regression to better fit the weights
- model to fit the weights for the Pott's model
