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

message = "file regression 1.7"

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


def fit_weights_potts_unregularized(X, y, T, L=L, A=A, sample_weight=None):
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

    sample_weight : optional (N,) per-observation weights -- solves the weighted
        least-squares problem by scaling each row of (X, y) by sqrt(w). None
        (default) reproduces the original unweighted OLS exactly.

    Returns
    -------
    F_hat, J_hat, rank : recovered tensors plus the numerical rank of X
        (rank < X.shape[1] flags a rank-deficient / non-unique OLS fit).
    """
    if sample_weight is None:
        Xw, yw = X, y
    else:
        sw = np.sqrt(np.asarray(sample_weight, dtype=np.float64))
        Xw, yw = sw[:, None] * X, sw * y
    w, _residuals, rank, _sv = np.linalg.lstsq(Xw, yw, rcond=None)
    F_hat, J_hat = _unpack_potts_weights(w, T, L=L, A=A)
    return F_hat, J_hat, rank


### ---------------------------- End-to-end pipeline --------------------------- ###
#######################################################################################

def fit_weights_potts_from_data(seq_matrix, target, sample_weight=None, lambdas_grid=None,
                                 k_folds=5, seed=0, verbose=True, lam=None):
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
    lam           : optional fixed L2 penalty. None (default) picks lambda by K-fold CV over
                    lambdas_grid (original behavior). lam=0 skips CV and does a minimum-norm
                    (SVD lstsq) unregularized fit -- use this when CV keeps landing on a huge
                    lambda that just shrinks everything to the mean. lam>0 skips CV and fits
                    at exactly that penalty. When lam is not None, info["cv_mse"] /
                    info["lambdas_grid"] are None.

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
        if verbose and lam is None:
            print(f"lambdas_grid was not defined thus lambdas_grid = {lambdas_grid}")

    X = build_potts_features(seq_matrix)
    y = np.asarray(target, dtype=np.float64)

    if lam == 0.0:
        # the fit below IS an lstsq -- do it once here and read rank off it, instead of
        # running a second (unweighted) lstsq just for the rank diagnostic.
        if verbose:
            print("lam=0 -> minimum-norm unregularized (lstsq) fit, CV skipped")
        F_hat, J_hat, rank = fit_weights_potts_unregularized(
            X, y, T=1.0, sample_weight=sample_weight)
        if verbose:
            print(f"Design matrix rank: {rank} / {X.shape[1]} features ({X.shape[0]} obs)")
        info = dict(lam=0.0, cv_mse=None, lambdas_grid=None, n_obs=X.shape[0])
        return F_hat, J_hat, rank, info

    _, _, rank = fit_weights_potts_unregularized(X, y, T=1.0)
    if verbose:
        print(f"Design matrix rank: {rank} / {X.shape[1]} features ({X.shape[0]} obs)")
        if rank < X.shape[1]:
            print("  NOTE: rank-deficient design (p > n) -- ridge is not just a preference, "
                  "the unregularized fit below is non-unique.")

    if lam is None:
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

    best_lam = float(lam)  # lam == 0.0 is handled by the early return above
    if verbose:
        print(f"lam={best_lam:.4g} (fixed) -> ridge fit, CV skipped")
    F_hat, J_hat = fit_weights_potts(X, y, T=1.0, lam=best_lam, sample_weight=sample_weight)

    info = dict(lam=best_lam, cv_mse=None, lambdas_grid=None, n_obs=X.shape[0])
    return F_hat, J_hat, rank, info


def fit_weights_glm_from_data(seq_matrix, target_ratio, power=1, alphas_grid=None,
                               k_folds=3, seed=0, max_iter=500, cv_subsample=25_000,
                               verbose=True):
    """
    Same Potts design (build_potts_features) and same F/J unpacking as
    fit_weights_potts_from_data, but fit with a Tweedie GLM (log link) on a NON-NEGATIVE
    enrichment RATIO target instead of ridge least squares on a log-enrichment target.

        E[ratio(s)] = exp( F.s + J.s.s + bias ),   Var[ratio(s)] proportional to E[.]**power

    power=1 -> Poisson deviance loss (variance proportional to mean). power=2 -> Gamma
    (constant coefficient of variation). power=0 would be ordinary least squares (use
    fit_weights_potts_from_data for that). The recovered F/J live on a log scale -- directly
    comparable to fit_weights_potts_from_data's F/J, since that fits log(ratio) directly and
    this fits log(E[ratio]).

    Caveat for power=1 (Poisson): the Poisson objective is scale-equivariant -- multiplying
    every target_ratio by a constant multiplies the whole deviance by that constant, leaving
    argmin F/J unchanged. So with only a log-enrichment column (no real before/after read
    counts to set a per-sequence offset), a Poisson fit on exp(log_enrichment) is entirely
    determined by the shape of the ratio distribution, and its loss is dominated by the few
    most-enriched sequences (largest ratio -> largest fitted mean -> largest IRLS weight).
    A genuinely count-aware Poisson GT needs the raw plasmid/vector counts (aav2.csv/aav5.csv
    expose these; aav9.csv does not).

    Solver / cost: uses lbfgs, NOT newton-cholesky. newton-cholesky forms the p x p Hessian
    (p = 8540 Potts features -> a 8540x8540 X^T W X every Newton step, ~1e12 flops each) and,
    at small alpha where the rank-deficient design (826 never-co-observed cells) makes that
    Hessian near-singular, grinds through max_iter steps -- a single fit can take an hour.
    lbfgs is O(n*p) per iteration (~1e8 flops), so a fit is seconds-to-a-minute regardless of
    alpha. CV is additionally run on a `cv_subsample`-row random subsample (alpha selection
    does not need all ~69k rows); the final refit uses the full dataset.

    seq_matrix   : (N, L) amino-acid indices, same convention as build_potts_features.
    target_ratio : (N,) strictly non-negative fold-enrichment ratio (e.g. exp(aav9 target),
                   or fit4functionaav9.csv's Production column). NOT a log enrichment.
    cv_subsample : rows to subsample for the CV alpha sweep (None -> use all). The final fit
                   always uses every row.

    Returns
    -------
    F_hat, J_hat, info
        info : dict with best alpha, its CV deviance curve, the alphas grid, n_obs, power.
        (No `rank` slot, unlike fit_weights_potts_from_data -- a GLM's iteratively reweighted
        design has no single fixed rank to report.)
    """
    from sklearn.linear_model import TweedieRegressor
    from sklearn.metrics import mean_tweedie_deviance

    target_ratio = np.asarray(target_ratio, dtype=np.float64)
    if np.any(target_ratio < 0):
        raise ValueError("target_ratio must be non-negative (it is a fold-enrichment ratio, "
                         "not a log enrichment)")

    if alphas_grid is None:
        # Reaches lower than the ridge grid (down to 1e-3): both the Poisson and Gamma CV
        # deviance on aav9 have their interior minimum around alpha 5e-3..3e-2, well below
        # the ridge MSE's ~24 -- a 0.1 floor cuts off the real optimum. lbfgs (not
        # newton-cholesky) makes these low alphas safe: it never forms the near-singular
        # p x p Hessian, the rank-deficient directions just sit at 0 with zero gradient.
        alphas_grid = np.logspace(-3, 2, 8)
        if verbose:
            print(f"alphas_grid was not defined thus alphas_grid = {alphas_grid}")

    def _fit(alpha, Xtr, ytr):
        m = TweedieRegressor(power=power, alpha=float(alpha), link="log",
                             fit_intercept=True, max_iter=max_iter, tol=1e-6, solver="lbfgs")
        m.fit(Xtr, ytr)
        return m

    X_full = build_potts_features(seq_matrix)
    X = X_full[:, :-1]  # drop build_potts_features' bias column; TweedieRegressor fits its own intercept

    if cv_subsample is not None and cv_subsample < X.shape[0]:
        sub = np.random.default_rng(seed).choice(X.shape[0], size=cv_subsample, replace=False)
        X_cv, y_cv = X[sub], target_ratio[sub]
    else:
        X_cv, y_cv = X, target_ratio

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    cv_dev = np.zeros(len(alphas_grid))
    for tr, va in tqdm(list(kf.split(X_cv)), desc="GLM CV", leave=False):
        for k, alpha in enumerate(alphas_grid):
            m = _fit(alpha, X_cv[tr], y_cv[tr])
            cv_dev[k] += mean_tweedie_deviance(y_cv[va], m.predict(X_cv[va]), power=power)
    cv_dev /= kf.get_n_splits()
    best_alpha = float(alphas_grid[np.argmin(cv_dev)])

    if verbose:
        fam = {1: "Poisson", 2: "Gamma"}.get(power, f"Tweedie(power={power})")
        print(f"[{fam}] best alpha: {best_alpha:.4g}  (CV on {X_cv.shape[0]:,} rows)")
        if best_alpha in (alphas_grid[0], alphas_grid[-1]):
            edge = "lower" if best_alpha == alphas_grid[0] else "upper"
            print(f"  WARNING: best alpha is at the {edge} grid boundary -- widen alphas_grid.")

    final = _fit(best_alpha, X, target_ratio)

    w = np.append(final.coef_, final.intercept_)  # [single-site | pairwise | bias] -- matches _unpack_potts_weights
    F_hat, J_hat = _unpack_potts_weights(w, T=1.0, L=L, A=A)

    info = dict(alpha=best_alpha, cv_deviance=cv_dev, alphas_grid=alphas_grid,
                n_obs=X.shape[0], n_cv=X_cv.shape[0], power=power,
                n_iter=int(np.max(final.n_iter_)), converged=bool(np.max(final.n_iter_) < max_iter))
    return F_hat, J_hat, info


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


### ---------------------- MLE multinomial fit (Fernandez-de-Cossio-Diaz et al.) -------- ###
#######################################################################################

def fit_weights_potts_mle_multinomial(seq_matrix, N0, N1, eps=0.5, lam=1.0, maxiter=300,
                                       theta0=None, verbose=True):
    """
    Fits F/J by maximum likelihood directly on RAW before/after read counts, following the
    "rare binding approximation" of Fernandez-de-Cossio-Diaz, Uguzzoni & Pagnani (2021,
    Mol. Biol. Evol. 38(1):318-328, doi:10.1093/molbev/msaa204), eqs. (1)-(3), restricted to
    the T=1 (single round, two sequenced time points) case -- exactly our AAV viability
    checkpoint structure (plasmid t=0 -> virus t=1), and exactly their own minimal/highest-
    coverage data set (Olson et al.: 1 selection round, 2 sequenced time points).

    Contrast with fit_weights_potts_from_data: that method first collapses (N0, N1) into a
    single noisy scalar target y(s) = log((N1+eps)/(N0+eps)) per sequence, then does weighted
    least squares on y(s) -- i.e. it assumes a Gaussian residual around an already-lossy
    summary statistic. This function instead writes the multinomial log-likelihood of
    observing the ACTUAL round-to-round count vector N1 given N0 and the model, treating the
    whole library as one competitive draw (every read at t=1 is "assigned" to sequences in
    proportion to N0(s) * exp(score(s)), summed to 1 over the WHOLE population passed in).
    This is the paper's central claim: fitting the counts directly (rather than a pointwise
    ratio) extracts more signal per read and is far less sensitive to decimation/low coverage
    (their fig. 2) -- exactly the regime AAV2's organoid CSV sits in (median compte_plasmide=1
    for the whole dataset).

    Model (rare-binding approximation, their eq. 3, T=1):
        p(s) = N0(s) * exp(score(s)) / sum_r [ N0(r) * exp(score(r)) ]     (softmax over the
                                                                             WHOLE population)
        NLL(F,J) = -sum_s N1(s) * [ log(N0(s)+eps) + score(s) - logsumexp_r(log(N0(r)+eps) +
                   score(r)) ]   +  lam * ||theta||^2

    score(s) = F[s_pos, pos].sum() + sum_{i<j} J[i,j,s_i,s_j] -- exactly score_FJ's
    convention elsewhere in this project (their fitness f(s) = -E(s), same sign).

    No bias term (unlike fit_weights_potts/fit_weights_potts_unregularized): a constant added
    to every sequence's score cancels exactly in the softmax normalization, so it is not
    identifiable here and is simply omitted (n_params = L*A + n_pairs*A*A, not +1).

    eps=0.5 pseudocount is applied to N0 (as the paper explicitly states: "we add a
    pseudo-count of 1/2 to all counts ... before carrying out the inference") -- and to N1
    only in the sense that N1=0 sequences still contribute their (zero) count validly to the
    softmax denominator through N0; N1 itself needs no pseudocount since it only ever appears
    as a multiplicative weight (N1(s)=0 simply drops that sequence's log-likelihood term).

    IMPORTANT: `seq_matrix`/`N0`/`N1` define the population the softmax competes over. For a
    held-out evaluation, fit on TRAIN sequences only (their N0/N1 restricted to that subset)
    -- do not pass the full dataset if you intend to score TEST sequences separately with the
    returned F/J (evaluation itself needs no logZ/softmax, since Pearson r is invariant to a
    constant per-sequence shift -- just call score_FJ-style scoring on the held-out set).

    Optimizer: L-BFGS-B via scipy.optimize.minimize (jac=True, gradient from jax.grad) --
    matches the paper's own choice ("we found that the L-BFGS algorithm performed well").
    The forward pass costs O(n_obs * n_pairs) (gather + sum, no dense design matrix ever
    materialized), so a full AAV2 organoid-CSV fit (~millions of rows) is tractable per
    iteration; wall-clock cost scales with maxiter.

    Parameters
    ----------
    seq_matrix : (N, L) int array, amino-acid indices (build_potts_features convention).
    N0         : (N,) raw "before" round read counts (e.g. compte_plasmide).
    N1         : (N,) raw "after" round read counts (e.g. compte_virus).
    eps        : pseudocount added to N0 before taking its log (paper's own convention,
                 matches this project's eps=0.5 standard for self-computed log enrichments).
    lam        : L2 penalty on the flat [F | J] parameter vector (no bias to exclude).
    maxiter    : max L-BFGS-B iterations.
    theta0     : optional (n_params,) warm start (e.g. from a prior fit at a nearby lam).
                 None -> zeros (uniform p(s) proportional to N0(s) alone, sensible neutral start).

    Returns
    -------
    F_hat, J_hat, info
        info : dict with scipy's success/nit/fun (final NLL)/message, and n_obs.
    """
    seq_matrix = np.asarray(seq_matrix)
    n_obs, Lseq = seq_matrix.shape
    n_pairs = Lseq * (Lseq - 1) // 2
    n_params = Lseq * A + n_pairs * A * A

    logN0 = jnp.asarray(np.log(np.asarray(N0, dtype=np.float64) + eps))
    N1j = jnp.asarray(np.asarray(N1, dtype=np.float64))
    seq_j = jnp.asarray(seq_matrix)
    pair_idx = [(i, j) for i in range(Lseq) for j in range(i + 1, Lseq)]

    def unpack(theta):
        F_flat = theta[:Lseq * A].reshape(Lseq, A)
        F = F_flat.T  # (A, L)
        J_flat = theta[Lseq * A:].reshape(n_pairs, A, A)
        return F, J_flat

    def score_fn(theta):
        F, J_flat = unpack(theta)
        s = F[seq_j, jnp.arange(Lseq)].sum(axis=1)
        for k, (i, j) in enumerate(pair_idx):
            s = s + J_flat[k][seq_j[:, i], seq_j[:, j]]
        return s

    def nll(theta):
        score = score_fn(theta)
        logits = logN0 + score
        logZ = jax.scipy.special.logsumexp(logits)
        ll = jnp.sum(N1j * (logits - logZ))
        reg = lam * jnp.sum(theta ** 2)
        return -ll + reg

    value_and_grad = jax.jit(jax.value_and_grad(nll))

    def scipy_obj(theta_np):
        theta_j = jnp.asarray(theta_np, dtype=jnp.float64)
        v, g = value_and_grad(theta_j)
        return float(v), np.asarray(g, dtype=np.float64)

    if theta0 is None:
        theta0 = np.zeros(n_params, dtype=np.float64)

    from scipy.optimize import minimize
    res = minimize(scipy_obj, theta0, method="L-BFGS-B", jac=True,
                    options=dict(maxiter=maxiter))

    F_flat = res.x[:Lseq * A].reshape(Lseq, A)
    F_hat = F_flat.T
    J_flat = res.x[Lseq * A:].reshape(n_pairs, A, A)
    J_hat = np.zeros((Lseq, Lseq, A, A))
    for k, (i, j) in enumerate(pair_idx):
        J_hat[i, j] = J_flat[k]
        J_hat[j, i] = J_flat[k].T

    info = dict(success=bool(res.success), nit=int(res.nit), fun=float(res.fun),
                message=str(res.message), n_obs=n_obs, theta=res.x)
    if verbose:
        print(f"[MLE multinomial] converged={info['success']}  nit={info['nit']}  "
              f"NLL={info['fun']:.4g}  n_obs={n_obs:,}")
    return jnp.array(F_hat), jnp.array(J_hat), info


### ---------------------- Matrix-free ridge (memory-scalable) -------------------- ###
#######################################################################################

def fit_weights_potts_ridge_matrixfree(seq_matrix, target, sample_weight=None, lam=1.0,
                                        maxiter=1000, theta0=None, verbose=True):
    """
    Same ridge objective as fit_weights_potts/fit_weights_potts_unregularized --
    0.5*sum(w*(Xtheta - y)^2) + 0.5*lam*||theta_no_bias||^2 -- solved WITHOUT ever
    materializing the dense design matrix X (N, 8541).

    Why: fit_weights_potts/fit_weights_potts_unregularized build X via build_potts_features
    (a dense float32 array) and solve either the normal equations (np.linalg.solve on
    X.T@X, ~O(N*p^2) to form) or lstsq (SVD). Both require X itself to exist in memory --
    measured empirically (2026-09-18) at ~5x X's own size in peak RSS (14.6 GiB at
    N=90,000, 40.1 GiB at N=250,000 -- consistent ratio), which caps N at roughly 700-750k
    rows on a 121 GiB machine, and ~136 GiB just for X alone at N=4.27M (the full AAV2
    organoid CSV) -- not tractable at all.

    This function instead applies X (and implicitly X^T, via autodiff) as a linear
    operator, exactly the same way fit_weights_potts_mle_multinomial avoids materializing
    X for its multinomial likelihood: the forward pass is the SAME score_FJ-style
    gather-based computation used everywhere else in this project (O(N) memory, no N x p
    matrix), and jax.grad's reverse-mode autodiff through that gather IS the matrix-free
    X^T @ r operation (a gather's adjoint is a scatter-add -- exactly what "sum the
    residual over every row where this indicator feature is 1" means for X^T @ r, the
    same bincount-based trick already used for the "surrogate Potts" fit in
    AAV9_cross_packaging_parameter_sweeps.ipynb, just obtained here via autodiff instead
    of hand-written bincounts). Solved by L-BFGS-B (scipy.optimize.minimize) -- for this
    convex quadratic objective, L-BFGS converges to the same optimum as a direct solve,
    just iteratively instead of via one matrix factorization.

    Unlike fit_weights_potts_mle_multinomial, a bias term IS included and fit here (theta's
    last entry, never penalized -- same convention as fit_weights_potts's
    `reg[np.arange(n_feat-1), ...] += lam`, which also excludes the last/bias column):
    the softmax normalization that makes bias unidentifiable in the multinomial model does
    not apply to a plain weighted-least-squares objective. score_FJ elsewhere in this
    project never adds this bias back in -- harmless for Pearson r (translation-invariant),
    which is how every notebook in this project evaluates fit quality, but keep in mind if
    ever comparing raw score VALUES against fit_weights_potts_unregularized's output (which
    silently drops its own bias coefficient the same way, via _unpack_potts_weights only
    reading the first L*A + n_pairs*A*A entries of its solved vector).

    lam=0 (unregularized): L-BFGS started from theta0=zeros on a consistent linear system
    tends toward the minimum-norm solution (the same property that makes CG's iterates on
    singular systems converge to the minimum-norm least-squares solution when starting
    from 0) -- expected to approximately match fit_weights_potts_unregularized's SVD-based
    minimum-norm solution in the rank-deficient (p>n) regime, but this is an empirical
    match to VERIFY, not a guarantee with the same numerical exactness as an SVD -- see
    AAV2_potts_ridge_matrixfree_validation.ipynb for the actual comparison.

    Parameters
    ----------
    seq_matrix    : (N, L) amino-acid indices (build_potts_features convention).
    target        : (N,) real-valued regression target (log enrichment).
    sample_weight : optional (N,) per-observation weights. None -> unweighted (all 1).
    lam           : L2 penalty (same semantics/scale as fit_weights_potts's `lam`;
                    lam=0 -> unregularized, matches fit_weights_potts_unregularized).
    maxiter       : max L-BFGS-B iterations.
    theta0        : optional (n_params,) warm start. None -> zeros.

    Returns
    -------
    F_hat, J_hat, bias_hat, info
        info : dict with scipy's success/nit/fun (final objective value)/message, n_obs.
    """
    seq_matrix = np.asarray(seq_matrix)
    n_obs, Lseq = seq_matrix.shape
    n_pairs = Lseq * (Lseq - 1) // 2
    n_params = Lseq * A + n_pairs * A * A + 1  # +1 bias

    y = jnp.asarray(np.asarray(target, dtype=np.float64))
    if sample_weight is None:
        w = jnp.ones(n_obs, dtype=jnp.float64)
    else:
        w = jnp.asarray(np.asarray(sample_weight, dtype=np.float64))
    seq_j = jnp.asarray(seq_matrix)
    pair_idx = [(i, j) for i in range(Lseq) for j in range(i + 1, Lseq)]

    def unpack(theta):
        F_flat = theta[:Lseq * A].reshape(Lseq, A)
        F = F_flat.T  # (A, L)
        J_flat = theta[Lseq * A: Lseq * A + n_pairs * A * A].reshape(n_pairs, A, A)
        bias = theta[-1]
        return F, J_flat, bias

    def score_fn(theta):
        F, J_flat, bias = unpack(theta)
        s = F[seq_j, jnp.arange(Lseq)].sum(axis=1)
        for k, (i, j) in enumerate(pair_idx):
            s = s + J_flat[k][seq_j[:, i], seq_j[:, j]]
        return s + bias

    def loss(theta):
        resid = score_fn(theta) - y
        data_term = 0.5 * jnp.sum(w * resid ** 2)
        reg_term = 0.5 * lam * jnp.sum(theta[:-1] ** 2)  # bias excluded, same as fit_weights_potts
        return data_term + reg_term

    value_and_grad = jax.jit(jax.value_and_grad(loss))

    def scipy_obj(theta_np):
        theta_j = jnp.asarray(theta_np, dtype=jnp.float64)
        v, g = value_and_grad(theta_j)
        return float(v), np.asarray(g, dtype=np.float64)

    if theta0 is None:
        theta0 = np.zeros(n_params, dtype=np.float64)

    from scipy.optimize import minimize
    res = minimize(scipy_obj, theta0, method="L-BFGS-B", jac=True, options=dict(maxiter=maxiter))

    F_flat = res.x[:Lseq * A].reshape(Lseq, A)
    F_hat = F_flat.T
    J_flat = res.x[Lseq * A: Lseq * A + n_pairs * A * A].reshape(n_pairs, A, A)
    J_hat = np.zeros((Lseq, Lseq, A, A))
    for k, (i, j) in enumerate(pair_idx):
        J_hat[i, j] = J_flat[k]
        J_hat[j, i] = J_flat[k].T
    bias_hat = float(res.x[-1])

    info = dict(success=bool(res.success), nit=int(res.nit), fun=float(res.fun),
                message=str(res.message), n_obs=n_obs)
    if verbose:
        print(f"[ridge matrix-free] converged={info['success']}  nit={info['nit']}  "
              f"obj={info['fun']:.6g}  bias={bias_hat:.4g}  n_obs={n_obs:,}")
    return jnp.array(F_hat), jnp.array(J_hat), bias_hat, info


### -------------------- Shared scorer + matrix-free drop-in for fit_weights_potts_from_data ------ ###
#######################################################################################

def score_potts(seq_matrix, F, J, bias=0.0):
    """
    Deterministic Potts score s(seq) = F[seq_pos, pos].sum() + sum_{i<j} J[i,j,seq_i,seq_j]
    (+ bias). Pure numpy, O(N*L^2), no design matrix -- the same computation every notebook in
    this project has hand-rolled locally as a `score_FJ` helper; centralized here so new code
    (starting with fit_weights_potts_from_data_matrixfree's CV loop below) can reuse it instead
    of redefining it per notebook. Not used by the fitting functions themselves -- those need a
    JAX-differentiable version, kept internal to fit_weights_potts_ridge_matrixfree/
    fit_weights_potts_mle_multinomial.
    """
    F, J = np.asarray(F), np.asarray(J)
    seq_matrix = np.asarray(seq_matrix)
    Lseq = seq_matrix.shape[1]
    s = F[seq_matrix, np.arange(Lseq)].sum(axis=1).astype(np.float64)
    for i in range(Lseq):
        for j in range(i + 1, Lseq):
            s = s + J[i, j, seq_matrix[:, i], seq_matrix[:, j]]
    return s + bias


def fit_weights_potts_from_data_matrixfree(seq_matrix, target, sample_weight=None, lambdas_grid=None,
                                            k_folds=5, seed=0, verbose=True, lam=None, maxiter=1000):
    """
    Drop-in matrix-free replacement for fit_weights_potts_from_data -- SAME call signature, SAME
    return shape (F_hat, J_hat, rank, info with info["lam"]/info["cv_mse"]/info["lambdas_grid"]/
    info["n_obs"]), so an existing call site switches by renaming the function alone.

    **This is the recommended default for Potts regression fitting in this project going
    forward (2026-09-18)** -- not a specialized/experimental variant. Validated to reproduce
    fit_weights_potts/fit_weights_potts_unregularized's results essentially exactly on real AAV2
    data (r(F)/r(J) > 0.999999, both regularized and unregularized -- see
    AAV2_potts_ridge_matrixfree_validation.ipynb), while never materializing the dense (N, 8541)
    design matrix that caps fit_weights_potts_unregularized/fit_weights_potts at roughly 700-750k
    rows in practice (measured: ~5x the design matrix's own size in peak RSS -- 14.6 GiB at
    N=90,000, 40.1 GiB at N=250,000, consistent ratio -- on a 121 GiB machine). On AAV2 organoid
    data (4.27M rows), the old function cannot run on the full dataset at all (~136 GiB for the
    design matrix alone); this one fits it in under 30s.

    Unlike fit_weights_potts_from_data's CV loop (ridge_cv_mse_potts builds a dense X per fold --
    itself memory-bound the same way as the final fit), K-fold CV here reuses the matrix-free
    primitive (fit_weights_potts_ridge_matrixfree) for every fold's fit AND scores validation MSE
    via score_potts (no matrix, ever) -- so CV-based fitting is matrix-free end to end, not just
    a single fixed-lambda fit.

    rank : always None (no SVD/lstsq step -- an iterative solver has no natural rank diagnostic,
           see fit_weights_potts_ridge_matrixfree's docstring). Callers that print or branch on
           `rank` should treat None as "not computed", not "full rank".

    Parameters
    ----------
    Identical to fit_weights_potts_from_data: seq_matrix, target, sample_weight, lambdas_grid,
    k_folds, seed, verbose, lam. Plus maxiter (L-BFGS-B iteration cap per fit, default 1000 --
    fit_weights_potts_ridge_matrixfree converges in a few hundred iterations even at millions of
    rows; raise it if a fit reports converged=False).

    Returns
    -------
    F_hat, J_hat, rank, info -- rank is always None; info additionally carries info["bias"] (the
    fitted bias term, not present in fit_weights_potts_from_data's info dict -- score_potts
    ignores it by default like every score_FJ helper in this project does, since Pearson r is
    translation-invariant; only relevant if ever comparing raw score VALUES, not correlations).
    """
    if lambdas_grid is None:
        lambdas_grid = np.logspace(-1, 2, 30)
        if verbose and lam is None:
            print(f"lambdas_grid was not defined thus lambdas_grid = {lambdas_grid}")

    seq_matrix = np.asarray(seq_matrix)
    y = np.asarray(target, dtype=np.float64)
    n_obs = seq_matrix.shape[0]
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    if lam is not None:
        best_lam = float(lam)
        if verbose:
            tag = "minimum-norm unregularized" if best_lam == 0.0 else f"lam={best_lam:.4g} (fixed)"
            print(f"{tag} -> matrix-free ridge fit, CV skipped")
        F_hat, J_hat, bias_hat, _fit_info = fit_weights_potts_ridge_matrixfree(
            seq_matrix, y, sample_weight=w, lam=best_lam, maxiter=maxiter, verbose=False)
        info = dict(lam=best_lam, cv_mse=None, lambdas_grid=None, n_obs=n_obs, bias=bias_hat)
        return F_hat, J_hat, None, info

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    cv_mse = np.zeros(len(lambdas_grid))
    for tr, va in tqdm(list(kf.split(seq_matrix)), desc="Ridge CV (matrix-free)", leave=False):
        w_tr = None if w is None else w[tr]
        for k, lam_k in enumerate(lambdas_grid):
            F_k, J_k, bias_k, _ = fit_weights_potts_ridge_matrixfree(
                seq_matrix[tr], y[tr], sample_weight=w_tr, lam=float(lam_k),
                maxiter=maxiter, verbose=False)
            pred_va = score_potts(seq_matrix[va], F_k, J_k, bias=bias_k)
            cv_mse[k] += np.mean((y[va] - pred_va) ** 2)
    cv_mse /= kf.get_n_splits()
    best_lam = float(lambdas_grid[np.argmin(cv_mse)])

    if verbose:
        print(f"Best lambda: {best_lam:.4f}")
        if best_lam in (lambdas_grid[0], lambdas_grid[-1]):
            edge = "lower" if best_lam == lambdas_grid[0] else "upper"
            print(f"  WARNING: best lambda is at the {edge} grid boundary ({best_lam:.4g}) "
                  f"-- the true optimum may lie outside lambdas_grid; widen it.")

    F_hat, J_hat, bias_hat, _fit_info = fit_weights_potts_ridge_matrixfree(
        seq_matrix, y, sample_weight=w, lam=best_lam, maxiter=maxiter, verbose=False)
    info = dict(lam=best_lam, cv_mse=cv_mse, lambdas_grid=lambdas_grid, n_obs=n_obs, bias=bias_hat)
    return F_hat, J_hat, None, info
