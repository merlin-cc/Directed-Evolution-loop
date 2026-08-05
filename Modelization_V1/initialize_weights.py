"""
Loads F_viab/J_viab from the AAV9 notebook's ProfileMLP recovery (`F_mlp`/`J_mlp` in
AAV9_profile_model.ipynb, sections 3 and 3b) instead of the purely synthetic
`initialize_random_weights` in sequence_classesV1.py -- so a Protocol/ProtocolV2 simulation
can be seeded with weights that reflect the REAL AAV9 capsid data, instead of arbitrary
random ones.

Also builds F_sel/J_sel to go alongside that real F_viab/J_viab, in three flavors that trade
off how much F_sel/J_sel's own trend (which amino acid works where) resembles F_viab/J_viab's:
initialize_correlated_weights, initialize_anticorrelated_weights and initialize_indep_weights.
"""
import os

import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from sequence_classesV1 import initialize_random_weights
from analysisV1 import AA_LABELS, pearson, precision_at_k, plot_teacher_weights, plot_topk_recovery

_HERE = os.path.dirname(os.path.abspath(__file__))

message = "File version 1.1"

# Exported by AAV9_profile_model.ipynb's "Export F_viab and J_viab" cell (right after
# `F_mlp_fj, J_mlp = extract_effective_FJ_mlp(...)`, section 3b). Kept as plain .npy files
# (gitignored) rather than checked into git, since they're derived artifacts, not source --
# regenerate them by re-running that notebook's cell whenever model_mlp is retrained.
AAV9_F_VIAB_PATH = os.path.join(_HERE, "aav9_F_viab_mlp.npy")
AAV9_J_VIAB_PATH = os.path.join(_HERE, "aav9_J_viab_mlp.npy")

NUM_AMINO_ACIDS = 20
NUM_POSITIONS   = 7


def load_F_viab_aav9_mlp(path=AAV9_F_VIAB_PATH):
    """
    Loads the (num_amino_acids=20, num_positions=7) viability profile recovered by
    ProfileMLP's single-mutant scan on the real AAV9 dataset -- same array as `F_mlp` in
    AAV9_profile_model.ipynb ('Modelization_V1/MLP_regression/AAV_viability_test/'), section 3.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run AAV9_profile_model.ipynb's export cell first "
            f"('Modelization_V1/MLP_regression/AAV_viability_test/AAV9_profile_model.ipynb', "
            f"section 3b, right after `F_mlp_fj, J_mlp = extract_effective_FJ_mlp(...)`)."
        )
    F_viab = np.load(path)
    expected_shape = (NUM_AMINO_ACIDS, NUM_POSITIONS)
    if F_viab.shape != expected_shape:
        raise ValueError(f"expected F_viab shape {expected_shape}, got {F_viab.shape}")
    return jnp.asarray(F_viab)


def load_J_viab_aav9_mlp(path=AAV9_J_VIAB_PATH):
    """
    Loads the (num_positions=7, num_positions=7, num_amino_acids=20, num_amino_acids=20)
    pairwise coupling recovered by ProfileMLP's double-mutant scan on the real AAV9 dataset
    -- same array as `J_mlp` in AAV9_profile_model.ipynb, section 3b, with the diagonal
    (i == j, undefined "self interaction") zeroed out at export time, matching
    sequence_classesV1.build_J's convention (Protocol's J tensors are never NaN).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run AAV9_profile_model.ipynb's export cell first "
            f"('Modelization_V1/MLP_regression/AAV_viability_test/AAV9_profile_model.ipynb', "
            f"section 3b, right after `F_mlp_fj, J_mlp = extract_effective_FJ_mlp(...)`)."
        )
    J_viab = np.load(path)
    expected_shape = (NUM_POSITIONS, NUM_POSITIONS, NUM_AMINO_ACIDS, NUM_AMINO_ACIDS)
    if J_viab.shape != expected_shape:
        raise ValueError(f"expected J_viab shape {expected_shape}, got {J_viab.shape}")
    return jnp.asarray(J_viab)


### ------------------- F_sel / J_sel built to a target trend vs F_viab / J_viab ------------------- ###
#######################################################################################################
# Three ways to build F_sel/J_sel to go alongside an already-known F_viab/J_viab (e.g. loaded from
# AAV9 above, or from initialize_random_weights), each trading off how much F_sel/J_sel's own trend
# resembles F_viab/J_viab's:
#   - initialize_correlated_weights    : shares the trend (an amino acid good for viability tends
#                                         to be good for selectivity too), strength set by `correlation`
#   - initialize_anticorrelated_weights: shares the OPPOSITE trend (good for viability -> bad for
#                                         selectivity), strength set by `anticorrelation`
#   - initialize_indep_weights         : no shared trend -- amino-acid identity relabeled + a bit of noise
#######################################################################################################

def _correlated_F(key, F_viab, correlation):
    """
    F_sel = mu + correlation * (F_viab - mu) + sqrt(1 - correlation**2) * sigma * noise, with
    mu/sigma F_viab's OWN mean/std and noise fresh N(0, 1) draws from `key`. Unlike mixing in
    standardized (zero-mean/unit-std) space, this keeps F_sel on F_viab's own scale at every
    correlation level (Var(F_sel) == Var(F_viab) always) AND collapses to F_sel == F_viab
    EXACTLY at correlation=1 (not just an affine rescaling of it, which would still have
    correlation 1 but different raw values -- Pearson r alone doesn't distinguish the two).
    In expectation corr(F_sel, F_viab) == correlation for any target in between.
    """
    F_viab = jnp.asarray(F_viab)
    mu, sigma = F_viab.mean(), F_viab.std()
    noise = jax.random.normal(key, shape=F_viab.shape)
    mix   = mu + correlation * (F_viab - mu) + jnp.sqrt(max(1.0 - correlation ** 2, 0.0)) * sigma * noise
    return mix.astype(jnp.float32)


def _correlated_J(key, J_viab, correlation):
    """
    Same recipe as _correlated_F (mixed on J_viab's own mean/scale, so correlation=1 gives
    J_sel == J_viab exactly), applied to J_viab's off-diagonal (i != j) entries only (the
    diagonal has no "interaction" to correlate) and mirrored back into a symmetric (L, L, A,
    A) tensor the same way sequence_classesV1.build_J is (J[i, j, a, b] == J[j, i, b, a]),
    diagonal left at exactly 0.
    """
    J_viab = np.asarray(J_viab, dtype=np.float64)
    L, A   = J_viab.shape[0], J_viab.shape[2]
    pairs  = [(i, j) for i in range(L) for j in range(i + 1, L)]

    upper_vals = np.stack([J_viab[i, j] for i, j in pairs])  # (n_pairs, A, A)
    mu, sigma  = upper_vals.mean(), upper_vals.std()

    noise = np.asarray(jax.random.normal(key, shape=upper_vals.shape))
    mix   = mu + correlation * (upper_vals - mu) + np.sqrt(max(1.0 - correlation ** 2, 0.0)) * sigma * noise

    J_sel = np.zeros((L, L, A, A), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        J_sel[i, j] = mix[k]
        J_sel[j, i] = mix[k].T
    return jnp.asarray(J_sel)


def initialize_correlated_weights(key, F_viab, J_viab, correlation=0.7):
    """
    Builds F_sel/J_sel that share F_viab/J_viab's per-cell trend -- e.g. if amino acid D
    works well for viability at some position, it also tends to work well for selectivity
    there -- with the strength of that sharing set by `correlation` (target Pearson
    correlation between F_sel and F_viab, and between J_sel and J_viab).

    key            : jax PRNG key -- the only source of randomness (F_sel/J_sel's noise
                     component), so different keys give different, independently reproducible
                     F_sel/J_sel for the SAME F_viab/J_viab/correlation.
    F_viab, J_viab : the reference viability weights to correlate against, e.g. from
                     load_F_viab_aav9_mlp()/load_J_viab_aav9_mlp(), or from
                     initialize_random_weights.
    correlation    : target Pearson correlation, in [-1, 1]. 0 gives F_sel/J_sel an
                     independent Gaussian draw with F_viab/J_viab's own mean/std (close to
                     initialize_indep_weights, but Gaussian noise instead of a reshuffle); 1
                     makes F_sel/J_sel EXACTLY equal to F_viab/J_viab (not just an affine
                     rescaling of it). For NEGATIVE correlation, either pass it directly here
                     or use initialize_anticorrelated_weights.

    Returns
    -------
    F_sel, J_sel : (num_amino_acids, num_positions) and (num_positions, num_positions,
    num_amino_acids, num_amino_acids), same shapes/conventions as F_viab/J_viab.
    """
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f"correlation must be in [-1, 1], got {correlation}")
    key_F, key_J = jax.random.split(key)
    F_sel = _correlated_F(key_F, F_viab, correlation)
    J_sel = _correlated_J(key_J, J_viab, correlation)
    return F_sel, J_sel


def initialize_anticorrelated_weights(key, F_viab, J_viab, anticorrelation=0.7):
    """
    Same as initialize_correlated_weights, but F_sel/J_sel share the OPPOSITE trend from
    F_viab/J_viab -- an amino acid that's great for viability tends to be BAD for
    selectivity, and vice versa. `anticorrelation` is a magnitude in [0, 1] (not a signed
    correlation): internally this just calls initialize_correlated_weights with
    correlation=-anticorrelation.
    """
    if not 0.0 <= anticorrelation <= 1.0:
        raise ValueError(
            f"anticorrelation must be in [0, 1] (a magnitude -- applied as a NEGATIVE "
            f"correlation internally), got {anticorrelation}"
        )
    return initialize_correlated_weights(key, F_viab, J_viab, correlation=-anticorrelation)


def initialize_indep_weights(key, F_viab, J_viab, noise_scale=0.2):
    """
    Builds F_sel/J_sel by RELABELING amino-acid identity: draw one random permutation sigma
    of the 20 amino acids and apply it to F_viab's amino-acid axis (its rows, axis 0) AND to
    BOTH of J_viab's amino-acid axes (axes 2 and 3) -- the SAME sigma for both, so the
    relabeling is coherent between the single-site and pairwise terms ("amino acid X" means
    the same thing in F_sel and J_sel). This keeps each amino acid's own position-dependent
    shape (and each pair's coupling shape) intact -- it's still a coherent profile/coupling
    model -- but reassigns WHICH amino acid gets which shape, so any per-cell correlation with
    F_viab/J_viab's ORIGINAL amino-acid identities is destroyed (an amino acid that's great
    for viability at position 3 has no reason to still be great for selectivity anywhere in
    particular).

    `jax.random.permutation(key, x, axis=1)` on F_viab (shape (A, L)) would permute axis 1
    (POSITIONS) instead -- with its default independent=False that reorders whole COLUMNS
    together, i.e. every amino acid's row keeps the exact same set of values, just in a
    different column order, so F_sel stays heavily correlated with F_viab. Permuting axis 0
    (the amino-acid rows) is what actually reassigns identity.

    A small amount of Gaussian noise (`noise_scale`, a fraction of F_viab's/J_viab's own std)
    is added on top afterward, so F_sel/J_sel reproduce the SHAPE of the profile/coupling
    weights without being a bit-exact relabeling -- pass noise_scale=0 for an exact relabeling.

    key            : jax PRNG key -- controls both the permutation and the noise, so a
                     different key gives a different (independently reproducible) F_sel/J_sel.
    F_viab, J_viab : the reference viability weights to relabel.
    noise_scale    : noise added on top, as a fraction of F_viab's (resp. J_viab's) own std
                     (default 0.2 -- "a bit of noise").
    """
    F_viab = jnp.asarray(F_viab)
    J_viab = jnp.asarray(J_viab)
    A, L   = F_viab.shape[0], J_viab.shape[0]

    key_perm, key_noise_F, key_noise_J = jax.random.split(key, 3)
    sigma = jax.random.permutation(key_perm, A)  # random relabeling of the 20 amino acids

    F_sel = F_viab[sigma, :]
    J_sel = J_viab[:, :, sigma, :][:, :, :, sigma]  # same sigma on both amino-acid axes

    if noise_scale > 0:
        F_sel = F_sel + noise_scale * F_viab.std() * jax.random.normal(key_noise_F, F_sel.shape)

        pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
        bumps = noise_scale * J_viab.std() * jax.random.normal(key_noise_J, (len(pairs), A, A))
        J_sel = np.array(J_sel)
        for k, (i, j) in enumerate(pairs):
            bump = np.asarray(bumps[k])
            J_sel[i, j] += bump   # off-diagonal only -- diagonal (i == j) stays exactly 0
            J_sel[j, i] += bump.T  # keep J[i, j, a, b] == J[j, i, b, a]

    return F_sel.astype(jnp.float32), jnp.asarray(J_sel, dtype=jnp.float32)


"""if __name__ == "__main__":
    F_viab = load_F_viab_aav9_mlp()
    J_viab = load_J_viab_aav9_mlp()

    key = jax.random.key(0)
    key_corr, key_anti, key_indep = jax.random.split(key, 3)

    F_corr, J_corr   = initialize_correlated_weights(key_corr, F_viab, J_viab, correlation=0.7)
    F_anti, J_anti   = initialize_anticorrelated_weights(key_anti, F_viab, J_viab, anticorrelation=0.7)
    F_indep, J_indep = initialize_indep_weights(key_indep, F_viab, J_viab)

    print("Achieved Pearson r(F_sel, F_viab) / r(J_sel, J_viab) -- sanity check that the")
    print("correlation/anticorrelation/indep parametrization behaves as intended:")
    off_diag = ~np.eye(NUM_POSITIONS, dtype=bool)
    for name, F, J in [("correlated (target r=+0.70)", F_corr, J_corr),
                        ("anticorrelated (target r=-0.70)", F_anti, J_anti),
                        ("indep (permutation, target r~0)", F_indep, J_indep)]:
        r_F = pearson(np.asarray(F_viab).ravel(), np.asarray(F).ravel())
        r_J = pearson(np.asarray(J_viab)[off_diag].ravel(), np.asarray(J)[off_diag].ravel())
        print(f"  {name:35s} r_F = {r_F:+.3f}   r_J = {r_J:+.3f}")

    for F, title in [(F_viab, "F_viab (real AAV9)"), (F_corr, "F_sel (correlated, r=+0.7)"),
                      (F_anti, "F_sel (anticorrelated, r=-0.7)"), (F_indep, "F_sel (indep, permuted)")]:
        plot_teacher_weights(F, title=title)
    plt.show()"""
