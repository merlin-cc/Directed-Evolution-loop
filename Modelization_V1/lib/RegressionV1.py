"""
Weight recovery for the Potts model (F, J) from ONLY the information a real
AAV directed-evolution experiment would have access to: NGS read counts.

In sequence_classesV1.py, a Protocol tracks two families of arrays per round:
  - lambda0, lambda1, lambda2, lambda3, lambda4  -- internal simulation ground
    truth (true molecule/capsid/cell counts). NOT observable in a real assay.
  - lambda0p, lambda2p, lambda3p                 -- NGS read counts at the 3
    checkpoints (pipetting -> PCR -> sequencing applied to lambda0, lambda2,
    lambda3 respectively). THIS is the only data a real protocol measures.

This module regresses F_viab/J_viab (from lambda0p -> lambda2p, the viability
step) and F_sel/J_sel (from lambda2p -> lambda3p, the selectivity step) using
Ridge regression on log-enrichment-ratio targets, following the same feature
convention as First_modelization/Potts_regression.ipynb (single-site one-hot
+ pairwise outer-product features).

Multiple rounds of directed evolution (protocol.N_loop_DE) can be pooled into
one larger training set: the log-ratio target for a sequence at a given round,
y(s) = log(reads_after(s) / reads_before(s)), is -- in expectation -- the same
fixed score-driven quantity every round (numerator and denominator come from
the same round's pool, so the round-to-round drift in overall pool composition
cancels out). Stacking several rounds' observations is therefore like
collecting repeated noisy measurements of the same underlying score: it
averages out Poisson/NB sampling noise, which is valuable given how sparse a
single round's reads are relative to a large library.

Caveat (does not go away with more rounds): diversity collapses over
successive N_loop_DE rounds as low-score variants get purged. Later rounds
concentrate reads onto fewer, already-enriched sequences, so they mostly
de-noise the survivors rather than recovering information about sequences
already lost in round 1. Pooling rounds helps precision on the retained
variants; it does not recover a sequence that dropped to zero abundance
early and never resurfaces.
"""

import numpy as np
import jax
import jax.numpy as jnp
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from sequence_classesV1 import *
from analysisV1 import pearson, precision_at_k

message = "file regression 1.4"

L = 7   # num_positions, matches Protocol.compute_score's hardcoded range(7)
A = 20  # num_amino_acids


### ---------------------------- Feature construction --------------------------- ###
#######################################################################################

def build_potts_features(seqs_obs, A=A):
    """
    Single-site one-hot + pairwise outer-product feature matrix.

    Parameters
    ----------
    seqs_obs : (N, L) int array/sequence -- observed sequences (can repeat
               across rounds; each row is treated as one training example)
    A        : int, alphabet size

    Returns
    -------
    X : (N, L*A + L*(L-1)//2 * A**2 + 1) float32 numpy array
        [ single-site (L*A) | pairwise (n_pairs*A*A) | bias (1) ]
    """
    seqs_obs = np.array(seqs_obs)
    N, Lseq = seqs_obs.shape

    oh = np.eye(A, dtype=np.float32)[seqs_obs]  # (N, L, A)
    X1 = oh.reshape(N, -1)                      # (N, L*A)

    pairs = []
    for i in range(Lseq):
        for j in range(i + 1, Lseq):
            pair_ij = oh[:, i, :, None] * oh[:, j, None, :]  # (N, A, A)
            pairs.append(pair_ij.reshape(N, A * A))
    X2 = np.concatenate(pairs, axis=1)  # (N, n_pairs * A*A)

    bias = np.ones((N, 1), dtype=np.float32)
    return np.concatenate([X1, X2, bias], axis=1)


### ---------------------------- Multi-round dataset --------------------------- ###
#######################################################################################

def build_multi_round_dataset(protocol, n_rounds, eps=0.5):
    """
    Runs n_rounds of directed evolution (protocol.N_loop_DE) and pools the
    per-round log-ratio regression targets into one training set, using only
    lambda0p / lambda2p / lambda3p (the observable NGS reads).

    Viability target (lambda0p -> lambda2p):
        y_viab(s) = log( (lambda2p(s) + eps) / (lambda0p_norm(s) + eps) )
    Selectivity target (lambda2p -> lambda3p):
        y_sel(s)  = log( (lambda3p(s) + eps) / (lambda2p_norm(s) + eps) )
    where *_norm rescales the "before" pool to the same total as the "after"
    pool, so the ratio isolates the score effect from overall pool-size drift.

    Only sequences with a nonzero "before" read count are kept for each step
    (a zero denominator makes the log-ratio undefined / infinitely negative).

    Returns
    -------
    seqs_viab, y_viab, seqs_sel, y_sel : concatenated (sequence, target) pairs
        across all n_rounds, ready for build_potts_features().
    """
    rounds    = protocol.N_loop_DE(n_rounds)
    sequences = np.array(protocol.sequence)

    seqs_viab_list, y_viab_list = [], []
    seqs_sel_list,  y_sel_list  = [], []

    for bio_row, ngs_row in tqdm(rounds, total=n_rounds, desc="Simulating DE rounds", leave=False):
        lambda0p, lambda2p, lambda3p = (np.array(a, dtype=float) for a in ngs_row)

        mask_v = lambda0p > 0
        if mask_v.any():
            l0p_norm = lambda0p / lambda0p.sum() * lambda2p.sum()
            y_v = np.log((lambda2p + eps) / (l0p_norm + eps))
            seqs_viab_list.append(sequences[mask_v])
            y_viab_list.append(y_v[mask_v])

        mask_s = lambda2p > 0
        if mask_s.any():
            l2p_norm = lambda2p / lambda2p.sum() * lambda3p.sum()
            y_s = np.log((lambda3p + eps) / (l2p_norm + eps))
            seqs_sel_list.append(sequences[mask_s])
            y_sel_list.append(y_s[mask_s])

    seqs_viab = np.concatenate(seqs_viab_list, axis=0)
    y_viab    = np.concatenate(y_viab_list,    axis=0)
    seqs_sel  = np.concatenate(seqs_sel_list,  axis=0)
    y_sel     = np.concatenate(y_sel_list,     axis=0)

    return seqs_viab, y_viab, seqs_sel, y_sel


### ---------------------------- Ridge fit + CV --------------------------- ###
#######################################################################################

def ridge_cv_mse_potts(X, y, lambdas, kf, sample_weight=None, desc="Ridge CV"):
    """
    K-fold CV MSE for Ridge regression on the Potts feature matrix, sweeping
    a grid of L2 penalties. The bias column (last column of X) is never
    penalized.

    sample_weight : optional (N,) per-observation weights -- solves the weighted normal
        equations (Xtr.T @ diag(w) @ Xtr + lam*I) beta = Xtr.T @ diag(w) @ ytr instead of the
        unweighted ones. None (default) reproduces the original unweighted behavior exactly.
        Validation MSE is left unweighted either way (CV should score plain predictive error,
        not the training weighting).
    """
    mse = np.zeros(len(lambdas))
    n_feat = X.shape[1]
    for tr, va in tqdm(list(kf.split(X)), desc=desc, leave=False):
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]
        if sample_weight is None:
            G   = Xtr.T @ Xtr
            rhs = Xtr.T @ ytr
        else:
            wtr = sample_weight[tr]
            G   = Xtr.T @ (wtr[:, None] * Xtr)
            rhs = Xtr.T @ (wtr * ytr)
        for k, lam in enumerate(lambdas):
            reg = G.copy()
            reg[np.arange(n_feat - 1), np.arange(n_feat - 1)] += lam
            w = np.linalg.solve(reg, rhs)
            mse[k] += np.mean((yva - Xva @ w) ** 2)
    return mse / kf.get_n_splits()


def _unpack_potts_weights(w, T, L=L, A=A):
    """
    Shared by fit_weights_potts and fit_weights_potts_unregularized: splits a
    flat weight vector -- laid out as [single-site | pairwise | bias], the
    same column order build_potts_features produces -- into (F_hat, J_hat)
    tensors rescaled by temperature T.
    """
    F_flat = w[:L * A].reshape(L, A)
    F_hat  = (F_flat).T  # -> (A, L)

    n_pairs = L * (L - 1) // 2
    J_flat  = w[L * A : L * A + n_pairs * A * A].reshape(n_pairs, A, A)
    J_hat   = np.zeros((L, L, A, A))
    k = 0
    for i in range(L):
        for j in range(i + 1, L):
            J_hat[i, j] = J_flat[k]
            J_hat[j, i] = J_flat[k].T
            k += 1

    return jnp.array(F_hat), jnp.array(J_hat)


def fit_weights_potts(X, y, T, L=L, A=A, lam=1.0, sample_weight=None):
    """
    Ridge fit of the pooled log-ratio targets, then unpack the flat weight
    vector back into (F_hat, J_hat) tensors rescaled by temperature T.

    sample_weight : optional (N,) per-observation weights, same weighted-normal-equations
        convention as ridge_cv_mse_potts. None (default) reproduces the original unweighted
        fit exactly.
    """
    n_feat = X.shape[1]
    if sample_weight is None:
        G   = X.T @ X
        rhs = X.T @ y
    else:
        G   = X.T @ (sample_weight[:, None] * X)
        rhs = X.T @ (sample_weight * y)
    G[np.arange(n_feat - 1), np.arange(n_feat - 1)] += lam
    w = np.linalg.solve(G, rhs)
    return _unpack_potts_weights(w, T, L=L, A=A)


def fit_weights_potts_unregularized(X, y, T, L=L, A=A):
    """
    Ordinary least squares (no L2 penalty) fit of the pooled log-ratio
    targets, then unpack the flat weight vector back into (F_hat, J_hat)
    tensors rescaled by temperature T.

    Uses np.linalg.lstsq (SVD-based pseudoinverse) rather than solving the
    normal equations directly: with 8541 Potts features, X is routinely
    p >> n (rank-deficient X^T X), and np.linalg.solve on a singular/near-
    singular G is exactly the blowup fit_weights_potts's lam floor exists to
    avoid. lstsq instead returns the minimum-norm least-squares solution,
    which is well-defined even when X^T X is singular.

    Returns
    -------
    F_hat, J_hat, rank : recovered tensors plus the numerical rank of X
        (rank < X.shape[1] flags a rank-deficient / non-unique OLS fit).
    """
    w, _residuals, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    F_hat, J_hat = _unpack_potts_weights(w, T, L=L, A=A)
    return F_hat, J_hat, rank


### ---------------------------- End-to-end pipeline --------------------------- ###
#######################################################################################

def fit_weights_potts_from_data(seq_matrix, target, sample_weight=None, lambdas_grid=None,
                                 k_folds=5, seed=0, verbose=True):
    """
    Ridge-fit F/J directly on an observed (sequence, target) dataset -- e.g. real aav9.csv
    (one row per sequence, target already a real log enrichment) -- instead of a Protocol
    simulated over multiple rounds. Reuses the exact same build_potts_features /
    ridge_cv_mse_potts / fit_weights_potts / fit_weights_potts_unregularized recipe as
    recover_weights_from_NGS, just skipping build_multi_round_dataset (which only makes sense
    for a simulated Protocol's per-round NGS reads): aav9.csv already IS one flat (seq,
    target) table, no round pooling needed.

    No temperature rescaling (unlike fit_weights_potts's `T` for a Protocol's T_viab/T_sel):
    `target` here is already a real log enrichment, not a Protocol score to be divided by a
    temperature, so T=1.0 is passed through and F_hat/J_hat come out on target's own scale.

    seq_matrix    : (N, L) raw amino-acid indices (same convention as build_potts_features).
    target        : (N,) real-valued regression target (e.g. aav9.csv's log-enrichment column).
    sample_weight : optional (N,) per-observation weights (e.g. 1/error**2 for datasets with a
                    real per-sequence uncertainty column, like aav2.csv/aav5.csv -- aav9.csv's
                    `error` is a constant, so leaving this None is equivalent there). When
                    given, both the CV loop and the final fit solve the weighted normal
                    equations (X.T @ diag(w) @ X + lam*I) beta = X.T @ diag(w) @ y instead of
                    the unweighted ones.

    Returns
    -------
    F_hat, J_hat, rank, info
        rank : numerical rank of the (unweighted) design matrix from the unregularized OLS
               fit, reported so callers can see whether they're in the n > p regime (real
               aav9.csv: ~68,776 rows vs 8,541 features) or the p >> n regime
               fit_weights_potts_unregularized's docstring warns about (small simulated
               multi-round datasets).
        info : dict with best lambda, its CV curve, and the lambdas grid.
    """
    if lambdas_grid is None:
        lambdas_grid = np.logspace(-1, 2, 30)
        if verbose:
            print(f"lambdas_grid was not defined thus lambdas_grid = {lambdas_grid}")

    X = build_potts_features(seq_matrix)
    y = np.asarray(target, dtype=np.float64)

    _, _, rank = fit_weights_potts_unregularized(X, y, T=1.0)
    if verbose:
        print(f"Design matrix rank: {rank} / {X.shape[1]} features ({X.shape[0]} obs)")
        if rank < X.shape[1]:
            print("  NOTE: rank-deficient design (p > n) -- ridge is not just a preference, "
                  "the unregularized fit below is non-unique.")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    cv_mse   = ridge_cv_mse_potts(X, y, lambdas_grid, kf, sample_weight=sample_weight,
                                   desc="Ridge CV (aav9)")
    best_lam = float(lambdas_grid[np.argmin(cv_mse)])

    if verbose:
        print(f"Best lambda: {best_lam:.4f}")
        if best_lam in (lambdas_grid[0], lambdas_grid[-1]):
            edge = "lower" if best_lam == lambdas_grid[0] else "upper"
            print(f"  WARNING: best lambda is at the {edge} grid boundary ({best_lam:.4g}) "
                  f"-- the true optimum may lie outside lambdas_grid; widen it.")

    F_hat, J_hat = fit_weights_potts(X, y, T=1.0, lam=best_lam, sample_weight=sample_weight)

    info = dict(lam=best_lam, cv_mse=cv_mse, lambdas_grid=lambdas_grid, n_obs=X.shape[0])
    return F_hat, J_hat, rank, info


def recover_weights_from_NGS(protocol, n_rounds=5, lambdas_grid=None, k_folds=5,
                              eps=0.5, seed=0, verbose=True):
    """
    End-to-end weight recovery using ONLY protocol's observable NGS reads
    (lambda0p, lambda2p, lambda3p) collected over n_rounds of directed
    evolution -- never lambda0..lambda4 ground truth.

    Returns
    -------
    F_viab_hat, J_viab_hat, F_sel_hat, J_sel_hat, info
        info : dict with best lambda per step, CV curves, and dataset sizes
    """
    if lambdas_grid is None:
        # Floor of 0.1 keeps every fold's (G + lam*I) solve well away from the
        # p>>n null-space blowup (8541 Potts features vs. a few thousand
        # observed sequences at best -- float32 roundoff on G's ~zero
        # eigenvalues gets divided by lam and explodes for lam below ~0.1).
        # Ceiling is generous since large lam is numerically safe (it can
        # only underfit, never blow up) -- see the boundary check below.
        lambdas_grid = np.logspace(-1, 2, 30)
        print(f"lambdas_grid was not defined thus lambdas_grid = {lambdas_grid}")

    if verbose:
        print(f"JAX backend: {jax.default_backend()} -- devices: {jax.devices()}")

    seqs_viab, y_viab, seqs_sel, y_sel = build_multi_round_dataset(protocol, n_rounds, eps=eps)

    if verbose:
        print(f"Viability   dataset: {seqs_viab.shape[0]} observed (sequence, round) pairs over {n_rounds} rounds")
        print(f"Selectivity dataset: {seqs_sel.shape[0]} observed (sequence, round) pairs over {n_rounds} rounds")

    X_viab = build_potts_features(seqs_viab)
    X_sel  = build_potts_features(seqs_sel)
    print("Just finished building the Potts features")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    cv_viab  = ridge_cv_mse_potts(X_viab, y_viab, lambdas_grid, kf, desc="CV (viability)")
    lam_viab = float(lambdas_grid[np.argmin(cv_viab)])

    cv_sel  = ridge_cv_mse_potts(X_sel, y_sel, lambdas_grid, kf, desc="CV (selectivity)")
    lam_sel = float(lambdas_grid[np.argmin(cv_sel)])

    if verbose:
        print(f"Best lambda (viability)   : {lam_viab:.4f}")
        print(f"Best lambda (selectivity) : {lam_sel:.4f}")
        for name, lam in (("viability", lam_viab), ("selectivity", lam_sel)):
            if lam in (lambdas_grid[0], lambdas_grid[-1]):
                edge = "lower" if lam == lambdas_grid[0] else "upper"
                print(f"  WARNING: best lambda ({name}) is at the {edge} grid boundary "
                      f"({lam:.4g}) -- the true optimum may lie outside lambdas_grid; widen it.")

    F_viab_hat, J_viab_hat = fit_weights_potts(X_viab, y_viab, protocol._T_viab, lam=lam_viab)
    F_sel_hat,  J_sel_hat  = fit_weights_potts(X_sel,  y_sel,  protocol._T_sel,  lam=lam_sel)

    info = dict(
        lam_viab=lam_viab, lam_sel=lam_sel,
        cv_mse_viab=cv_viab, cv_mse_sel=cv_sel,
        lambdas_grid=lambdas_grid,
        n_obs_viab=X_viab.shape[0], n_obs_sel=X_sel.shape[0],
    )
    return F_viab_hat, J_viab_hat, F_sel_hat, J_sel_hat, info


def recover_weights_unregularized_from_NGS(protocol, n_rounds=5, eps=0.5, verbose=True):
    """
    Same end-to-end pipeline as recover_weights_from_NGS (builds the pooled
    NGS dataset, then fits F_viab/J_viab and F_sel/J_sel), but with ordinary
    least squares instead of Ridge: no L2 penalty, so no lambda grid / CV
    step either. Useful as a no-regularization baseline against
    recover_weights_from_NGS, especially to see what the ridge penalty buys
    once the dataset is small relative to the 8541 Potts features (p >> n).

    Returns
    -------
    F_viab_hat, J_viab_hat, F_sel_hat, J_sel_hat, info
        info : dict with dataset sizes and the numerical rank of each design
        matrix (rank < n_features flags a rank-deficient / non-unique fit).
    """
    if verbose:
        print(f"JAX backend: {jax.default_backend()} -- devices: {jax.devices()}")

    seqs_viab, y_viab, seqs_sel, y_sel = build_multi_round_dataset(protocol, n_rounds, eps=eps)

    if verbose:
        print(f"Viability   dataset: {seqs_viab.shape[0]} observed (sequence, round) pairs over {n_rounds} rounds")
        print(f"Selectivity dataset: {seqs_sel.shape[0]} observed (sequence, round) pairs over {n_rounds} rounds")

    X_viab = build_potts_features(seqs_viab)
    X_sel  = build_potts_features(seqs_sel)
    print("Just finished building the Potts features")

    F_viab_hat, J_viab_hat, rank_viab = fit_weights_potts_unregularized(X_viab, y_viab, protocol._T_viab)
    F_sel_hat,  J_sel_hat,  rank_sel  = fit_weights_potts_unregularized(X_sel,  y_sel,  protocol._T_sel)

    if verbose:
        print(f"Viability   design matrix rank: {rank_viab} / {X_viab.shape[1]} features ({X_viab.shape[0]} obs)")
        print(f"Selectivity design matrix rank: {rank_sel} / {X_sel.shape[1]} features ({X_sel.shape[0]} obs)")
        for name, rank, X in (("viability", rank_viab, X_viab), ("selectivity", rank_sel, X_sel)):
            if rank < X.shape[1]:
                print(f"  WARNING: {name} design matrix is rank-deficient (p > n) -- lstsq returns "
                      f"the minimum-norm solution, not a unique OLS estimate.")

    info = dict(
        n_obs_viab=X_viab.shape[0], n_obs_sel=X_sel.shape[0],
        n_features=X_viab.shape[1],
        rank_viab=rank_viab, rank_sel=rank_sel,
    )
    return F_viab_hat, J_viab_hat, F_sel_hat, J_sel_hat, info


def evaluate_recovery(protocol, F_viab_hat, J_viab_hat, F_sel_hat, J_sel_hat):
    """
    Compares recovered weights/scores against the simulation's ground-truth
    F_viab/J_viab/F_sel/J_sel. Only usable in this simulated setting -- a
    real experiment has no such ground truth to check against.
    """
    v_gt  = np.array(protocol.compute_score(protocol.F_viab, protocol.J_viab))
    s_gt  = np.array(protocol.compute_score(protocol.F_sel,  protocol.J_sel))
    v_hat = np.array(protocol.compute_score(F_viab_hat, J_viab_hat))
    s_hat = np.array(protocol.compute_score(F_sel_hat,  J_sel_hat))

    combined_gt  = v_gt  / protocol._T_viab + s_gt  / protocol._T_sel
    combined_hat = v_hat / protocol._T_viab + s_hat / protocol._T_sel

    return dict(
        r_viab_scores      = pearson(v_gt, v_hat),
        r_sel_scores        = pearson(s_gt, s_hat),
        r_viab_weights      = pearson(np.array(protocol.F_viab).ravel(), np.array(F_viab_hat).ravel()),
        r_sel_weights       = pearson(np.array(protocol.F_sel).ravel(),  np.array(F_sel_hat).ravel()),
        precision_at_1pct   = precision_at_k(combined_gt, combined_hat, k_frac=0.01),
        precision_at_10pct  = precision_at_k(combined_gt, combined_hat, k_frac=0.10),
    )
