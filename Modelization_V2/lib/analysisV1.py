import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sequence_classesV1 import *

message = "file analysis 2.3"

# Standard 20 amino acid one-letter codes, alphabetical order. The alphabet
# indices (0..A-1) used elsewhere in this project carry no biological identity
# of their own -- this is purely a display convention for axis labels.
AA_LABELS = list("ACDEFGHIKLMNPQRSTVWY")


### ---------------------------- Mathematical analysis --------------------------- ###
#######################################################################################
#######################################################################################

def pearson(a, b):
    """
    corrcoef(a,b) = [[corr(a,a)  corr(a,b)]  =  [[1  r]
                    [corr(b,a)  corr(b,b)]]     [r  1]]
    """
    a, b = np.array(a), np.array(b)
    return float(np.corrcoef(a, b)[0, 1])

def precision_at_k(score_gt, score_hat, k_frac=0.10):
    """Fraction of top-k% by GT that are also in top-k% by predicted score."""
    N   = len(score_gt)
    k   = int(N * k_frac)
    top_gt  = set(np.argsort(-np.array(score_gt))[:k])
    top_hat = set(np.argsort(-np.array(score_hat))[:k])
    return len(top_gt & top_hat) / k

def topk_recovery(score_a, score_b, k=1000):
    """Fraction of the absolute top-k (by score_a) that are also in the absolute top-k (by
    score_b) -- same set-intersection-over-k recovery as precision_at_k, but parameterized by
    an absolute count instead of a population fraction, and symmetric: neither argument has to
    be the "ground truth" (e.g. useful to compare GT-vs-protocol, GT-vs-MLP and
    protocol-vs-MLP top-k sets against each other on the same footing)."""
    score_a, score_b = np.asarray(score_a), np.asarray(score_b)
    k = min(k, len(score_a))
    top_a = set(np.argsort(-score_a)[:k])
    top_b = set(np.argsort(-score_b)[:k])
    return len(top_a & top_b) / k

def shannon_entropy(counts):
    """Shannon entropy (nats) of a count vector, ignoring zero entries."""
    counts = np.array(counts, dtype=float)
    total  = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p)))

def count_over_threshold(counts, threshold=1.0):
    """Number of sequences whose count is >= threshold."""
    return int(np.sum(np.array(counts) >= threshold))

def _resolve_named_arrays(lambdas, names):
    """Shared helper: accept either a dict {name: array} or a plain list/tuple of
    arrays (+ optional names list), return (row_names, list_of_arrays)."""
    if isinstance(lambdas, dict):
        return list(lambdas.keys()), list(lambdas.values())
    arrays = list(lambdas)
    row_names = list(names) if names is not None else [f"seq_{i}" for i in range(len(arrays))]
    return row_names, arrays

def number_of_seq_threshold(lambdas, thresholds, names=None):
    """
    Table of the number of sequences with a count >= threshold, for several lambdas
    and thresholds -- one row per lambda, one column per threshold.

    lambdas    : either a dict {name: array} (e.g. {"lambda0": protocol.lambda0, ...}),
                 or a plain list/tuple/2D array of arrays -- in that case pass `names`
                 too (a name can't be recovered from a bare array value), e.g.:
                     number_of_seq_threshold(
                         lambdas=[lambda0, lambda1, lambda2, lambda3, lambda4],
                         names=["lambda0", "lambda1", "lambda2", "lambda3", "lambda4"],
                         thresholds=threshold_values,
                     )
                 If `names` is omitted for a non-dict input, rows are labelled by
                 position ("seq_0", "seq_1", ...) instead of guessing lambda names.
    thresholds : list of thresholds to check, e.g. [1, 10, 100, 1000]
    names      : optional list of row labels, only used when `lambdas` isn't a dict

    Returns
    -------
    pandas.DataFrame, rows = lambda names, columns = thresholds, values = count_over_threshold
    """
    if lambdas is None or thresholds is None:
        return None

    row_names, arrays = _resolve_named_arrays(lambdas, names)
    thresholds = [float(t) for t in thresholds]  # jax/numpy scalars aren't hashable as dict/column keys

    table = pd.DataFrame(
        {threshold: [count_over_threshold(arr, threshold) for arr in arrays]
         for threshold in thresholds},
        index=row_names,
    )
    table.index.name = "lambda"
    return table


def proportion_above_threshold(lambdas, thresholds_pct, names=None):
    """
    Table of how many sequences have a relative abundance above a given percentage
    of the total pool -- one row per lambda, one column per threshold.

    lambdas        : either a dict {name: array} (e.g. {"lambda0": protocol.lambda0,
                     ...}), or a plain list/tuple of arrays -- pass `names` too in
                     that case (same convention as number_of_seq_threshold)
    thresholds_pct : list of percentages to check, e.g. [0.1, 0.01, 0.001] means
                     "present at more than 0.1% / 0.01% / 0.001% of the pool total"
    names          : optional list of row labels, only used when `lambdas` isn't a dict

    Returns
    -------
    pandas.DataFrame, rows = lambda names, columns = "> X%" thresholds, values =
    number of sequences with count/sum(count) > threshold_pct/100
    """
    if lambdas is None or thresholds_pct is None:
        return None

    row_names, arrays = _resolve_named_arrays(lambdas, names)
    thresholds_pct = [float(t) for t in thresholds_pct]

    def _count_abundant(arr, pct):
        arr = np.array(arr, dtype=float)
        total = arr.sum()
        if total <= 0:
            return 0
        proportions = arr / total
        return int(np.sum(proportions > pct / 100.0))

    table = pd.DataFrame(
        {f"> {pct:g}%": [_count_abundant(arr, pct) for arr in arrays] for pct in thresholds_pct},
        index=row_names,
    )
    table.index.name = "lambda"
    return table

#######################################################################################
#######################################################################################
#######################################################################################




### ------------------------------- Curves analysis ------------------------------- ###
# To analyse the results multiple curves can be plotted, among them :
#   1 - Plotting the weight matrix of the Teacher model
#   2 - Plotting the initial library according the viability and selectivity score
#   3 - Count of each sequence after the viability and selectivity process
#   4 - The constructed library for next loop, with parameter N (number of loop)
#   5 - Plotting the 5-fold CV
#   6 - Matrix predicted by Student compared to the teacher one
#   7 - Top k% recovery
#   8 - plotting all lambdas in a loop
#   9 - Entropy vs sequencing depth: stochastic NGS reads vs the noise-free true pool
#  11 - Heatmap: top-k recovery across a 2D (N1, rho) parameter sweep
#######################################################################################
#######################################################################################

# 1 - Weight matrix of the Teacher model
def plot_teacher_weights(F, J=None, title="Teacher model weights"):
    """
    F : (num_amino_acids, L) profile matrix (e.g. protocol.F_viab or protocol.F_sel)
    J : (L, L, num_amino_acids, num_amino_acids) optional pairwise interaction matrix
        (e.g. protocol.J_viab or protocol.J_sel); if given, plotted as a per-position-pair
        coupling-strength heatmap alongside F
    """
    F = np.array(F)
    vmax = np.abs(F).max() or 1.0
    A, L = F.shape
    aa_labels = AA_LABELS[:A]

    if J is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(F, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('Position')
        ax.set_ylabel('Amino acid')
        ax.set_xticks(range(L))
        ax.set_xticklabels([str(i) for i in range(1, L + 1)])
        ax.set_yticks(range(A))
        ax.set_yticklabels(aa_labels)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label='Weight')
        fig.tight_layout()
        return fig

    J = np.array(J)
    j_strength = np.mean(J, axis=(2, 3))  # (L, L) coupling strength per position pair

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(F, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Amino acid')
    axes[0].set_xticks(range(L))
    axes[0].set_xticklabels([str(i) for i in range(1, L + 1)])
    axes[0].set_yticks(range(A))
    axes[0].set_yticklabels(aa_labels)
    axes[0].set_title('F (profile)')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(j_strength, cmap='viridis')
    axes[1].set_xlabel('Position j')
    axes[1].set_ylabel('Position i')
    axes[1].set_xticks(range(L))
    axes[1].set_xticklabels([str(i) for i in range(1, L + 1)])
    axes[1].set_yticks(range(L))
    axes[1].set_yticklabels([str(i) for i in range(1, L + 1)])
    axes[1].set_title('||J_ij|| coupling strength')
    fig.colorbar(im1, ax=axes[1])

    fig.suptitle(title)
    fig.tight_layout()
    return fig


# 2 - Initial library according to viability and selectivity score
def plot_initial_library_scores(protocol, title="Initial library: viability vs selectivity score"):
    """
    protocol : a sequence_classesV1.Protocol instance. Scores are computed with
               protocol.compute_score on protocol.sequence, and points are
               colored by protocol.lambda0 (initial abundance).
    """
    viab_scores = np.array(protocol.compute_score(protocol.F_viab, protocol.J_viab))
    sel_scores  = np.array(protocol.compute_score(protocol.F_sel, protocol.J_sel))
    counts      = np.array(protocol.lambda0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sca = ax.scatter(viab_scores, sel_scores, c=counts, s=20, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Viability score')
    ax.set_ylabel('Selectivity score')
    ax.set_title(title)
    fig.colorbar(sca, ax=ax, label='lambda0 (initial count)')
    fig.tight_layout()
    return fig


# 3 - Count of each sequence after the viability and selectivity process
def plot_counts_after_selection(protocol, title="Sequence counts through viability + selectivity"):
    """
    3 scatter panels: x = viability score (viab/T_viab), y = count at that stage
    (lambda2, lambda3, lambda4), color = selectivity score (sel/T_sel) — the two
    scores are kept on separate visual channels instead of being summed into one
    axis, so both are readable per point. Same x-axis and color scale across all
    3 panels. Each panel's legend reports how many variants are still findable
    (non-zero) at that stage.

    protocol : a Protocol instance that has already run a full round (e.g. via
               loop_DE()), so lambda2, lambda3 and lambda4 hold real values.
    """
    viab_scores = np.array(protocol.compute_score(protocol.F_viab, protocol.J_viab))
    sel_scores  = np.array(protocol.compute_score(protocol.F_sel, protocol.J_sel))
    x           = viab_scores / protocol._T_viab
    c           = sel_scores / protocol._T_sel
    vmin, vmax  = float(c.min()), float(c.max())

    steps = [
        ('lambda2', protocol.lambda2, 'post-production (viability)'),
        ('lambda3', protocol.lambda3, 'post-selectivity'),
        ('lambda4', protocol.lambda4, 'final library (post-bacterial amplification)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, y_arr, label) in zip(axes, steps):
        y = np.array(y_arr)
        mask = y > 0
        n_found = int(mask.sum())
        if mask.any():
            sca = ax.scatter(x[mask], y[mask], c=c[mask], cmap='RdYlGn',
                              vmin=vmin, vmax=vmax, alpha=1, s=45,
                              edgecolors='black', linewidths=0.6,
                              label=f'{n_found} variants findable')
            ax.set_yscale('log')
            plt.colorbar(sca, ax=ax, label='Selectivity score (sel/T_sel)', fraction=0.046, pad=0.04)
            ax.legend(loc='upper left', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
        ax.set_xlabel('Viability score (viab/T_viab)')
        ax.set_ylabel(name)
        ax.set_title(label)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# 4 - The constructed library for next loop, with parameter N (number of loop)
def plot_library_evolution(protocol, N, threshold=1e3):
    """
    Runs N rounds via protocol.N_loop_DE(N) (which now shows its own tqdm progress
    bar directly in sequence_classesV1.py) and plots Shannon entropy + sequence
    diversity of lambda0 (the round's starting library) across N directed-evolution
    rounds.

    Returns
    -------
    fig, rounds : the matplotlib figure and the raw list of [bio_row, ngs_row] pairs
                   (one per round, same format as N_loop_DE's return value), in case
                   further analysis of the run is needed.
    """
    rounds = protocol.N_loop_DE(N)

    H_list    = []
    diversity = []
    for bio_row, ngs_row in rounds:
        lambda0_round = bio_row[0]
        H_list.append(shannon_entropy(lambda0_round))
        diversity.append(count_over_threshold(lambda0_round, threshold))

    x = np.arange(1, N + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(x, H_list, marker='o', linestyle='-', linewidth=2, color='tab:blue')
    ax1.set_xlabel('Directed-evolution round')
    ax1.set_ylabel('Shannon entropy', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, diversity, marker='s', linestyle='--', linewidth=2, color='tab:red')
    ax2.set_ylabel(rf'Number of sequences with abundance $>{threshold:g}$', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.set_title(f'Library evolution over {N} directed-evolution rounds')
    fig.tight_layout()
    return fig, rounds


# 5 - 5-fold CV curve
def plot_cv_curve(lambdas_grid, mse_mean, mse_std=None, title="5-fold CV: MSE vs regularization"):
    """
    lambdas_grid : array of ridge regularization strengths (x-axis, log scale)
    mse_mean     : mean CV MSE for each entry of lambdas_grid
    mse_std      : optional std across the 5 folds, plotted as a shaded band
    """
    lambdas_grid = np.array(lambdas_grid)
    mse_mean     = np.array(mse_mean)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lambdas_grid, mse_mean, marker='o', color='tab:blue')
    if mse_std is not None:
        mse_std = np.array(mse_std)
        ax.fill_between(lambdas_grid, mse_mean - mse_std, mse_mean + mse_std,
                         alpha=0.2, color='tab:blue')

    best_idx = int(np.argmin(mse_mean))
    ax.axvline(lambdas_grid[best_idx], color='tab:red', linestyle='--', alpha=0.6,
               label=f'best λ = {lambdas_grid[best_idx]:.3g}')

    ax.set_xscale('log')
    ax.set_xlabel('Regularization strength (λ)')
    ax.set_ylabel('CV MSE')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# 6 - Matrix predicted by Student compared to the Teacher one
def plot_teacher_vs_student(F_teacher, F_student, title="Teacher vs Student weights"):
    """
    F_teacher : (num_amino_acids, L) ground-truth profile matrix (e.g. protocol.F_viab)
    F_student : (num_amino_acids, L) recovered/fitted profile matrix
    """
    F_teacher = np.array(F_teacher)
    F_student = np.array(F_student)
    vmax = max(np.abs(F_teacher).max(), np.abs(F_student).max()) or 1.0

    A, L = F_teacher.shape
    aa_labels = AA_LABELS[:A]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(F_teacher, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Teacher (ground truth)')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Amino acid')
    axes[0].set_xticks(range(L))
    axes[0].set_xticklabels([str(i) for i in range(1, L + 1)])
    axes[0].set_yticks(range(A))
    axes[0].set_yticklabels(aa_labels)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(F_student, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Student (recovered)')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Amino acid')
    axes[1].set_xticks(range(L))
    axes[1].set_xticklabels([str(i) for i in range(1, L + 1)])
    axes[1].set_yticks(range(A))
    axes[1].set_yticklabels(aa_labels)
    fig.colorbar(im1, ax=axes[1])

    r = pearson(F_teacher.ravel(), F_student.ravel())
    axes[2].scatter(F_teacher.ravel(), F_student.ravel(), s=8, alpha=0.4)
    lims = [min(F_teacher.min(), F_student.min()), max(F_teacher.max(), F_student.max())]
    axes[2].plot(lims, lims, 'k--', alpha=0.5)
    axes[2].set_xlabel('Teacher weight')
    axes[2].set_ylabel('Student weight')
    axes[2].set_title(f'Pearson r = {r:.3f}')

    fig.suptitle(title)
    fig.tight_layout()
    return fig


# 6b - Teacher vs Student SCORE (profile F + potts J combined), same scatter style as
# plot_teacher_vs_student's third panel above, but for the actual compute_score(F, J)
# output instead of raw F weights -- panel 3 above only checks single-position (F)
# recovery, this checks whether predictions track the combined single- + pairwise-
# interaction score, i.e. whether the model is doing well on BOTH interaction types.
def plot_score_teacher_vs_student(score_gt, score_hat, title=None):
    """
    score_gt, score_hat : (num_sequences,) scores from protocol.compute_score(F, J),
                          teacher vs recovered, e.g.
                              score_gt  = protocol.compute_score(protocol.F_viab, protocol.J_viab)
                              score_hat = protocol.compute_score(F_viab_hat, J_viab_hat)
                          or the T-scaled combined_gt/combined_hat (viability + selectivity)
                          from evaluate_profile_recovery, if checking both scores at once.
    """
    score_gt  = np.array(score_gt)
    score_hat = np.array(score_hat)

    r = pearson(score_gt, score_hat)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(score_gt, score_hat, s=8, alpha=0.4)
    lims = [min(score_gt.min(), score_hat.min()), max(score_gt.max(), score_hat.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('Teacher score (F + J)')
    ax.set_ylabel('Student score (F + J)')
    ax.set_title(title or f'Teacher vs Student score (profile + potts)   Pearson r = {r:.3f}')
    fig.tight_layout()
    return fig


# 7 - Top k% recovery
def plot_topk_recovery(score_gt, score_hat, k_frac=0.10, title=None,
                        xlabel='GT score', ylabel='Predicted score', ax=None):
    """
    GT-vs-predicted score scatter, colored by top-k% recovery category — matches
    the top-k recovery plot from Ridge_regression_for_linear_model.ipynb (cell 21):
    Recovered (top-k in both), Missed (GT top-k only), False positive
    (predicted top-k only), Rest — with the GT/predicted top-k thresholds drawn in.

    score_gt, score_hat : (num_sequences,) two scores to compare -- named "gt"/"hat" for the
                          common ground-truth-vs-predicted case, but neither has to actually be
                          a ground truth (e.g. protocol-measured vs. MLP-predicted both work).
    k_frac               : top fraction of the library defining "top-k" (default 10%)
    xlabel, ylabel       : axis labels, override when score_gt/score_hat aren't literally GT/predicted
    ax                   : existing Axes to draw into (e.g. one panel of a comparison grid) --
                          when given, no new Figure is created and the caller owns tight_layout()/
                          show(); when None (default), a standalone Figure is created and returned.
    """
    score_gt  = np.array(score_gt)
    score_hat = np.array(score_hat)
    N = len(score_gt)
    k = int(N * k_frac)

    top_gt   = set(np.argsort(-score_gt)[:k])
    top_hat  = set(np.argsort(-score_hat)[:k])
    both     = np.array(sorted(top_gt & top_hat))
    only_gt  = np.array(sorted(top_gt - top_hat))
    only_hat = np.array(sorted(top_hat - top_gt))
    rest     = np.array(sorted(set(range(N)) - top_gt - top_hat))

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(score_gt[rest],     score_hat[rest],     s=4,  alpha=0.2, color='lightgray',   label='Rest')
    ax.scatter(score_gt[only_hat], score_hat[only_hat], s=12, alpha=0.6, color='darkorange',  label=f'False pos. ({len(only_hat)})')
    ax.scatter(score_gt[only_gt],  score_hat[only_gt],  s=12, alpha=0.6, color='steelblue',   label=f'Missed ({len(only_gt)})')
    ax.scatter(score_gt[both],     score_hat[both],     s=12, alpha=0.8, color='forestgreen', label=f'Recovered ({len(both)})')

    lo = float(min(score_gt.min(), score_hat.min()))
    hi = float(max(score_gt.max(), score_hat.max()))
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)

    thr_gt  = float(np.sort(score_gt)[-k])
    thr_hat = float(np.sort(score_hat)[-k])
    ax.axvline(thr_gt,  color='steelblue',  linestyle=':', lw=1.2, label='GT threshold')
    ax.axhline(thr_hat, color='darkorange', linestyle=':', lw=1.2, label='Hat threshold')

    p = precision_at_k(score_gt, score_hat, k_frac=k_frac)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f'Top-{int(k_frac * 100)}% recovery   Precision = {p:.3f}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, markerscale=2, borderaxespad=0.3, borderpad=0.4, handlelength=1.2)
    if fig is not None:
        fig.tight_layout()
    return fig

# 8 - Plotting all lambdas for a loop
def plot_loop_lambdas(protocol, run_loop=True, title="All lambdas for one DE round"):
    """
    2x4 grid, one scatter panel per lambda produced by a single loop_DE() round.
    x = viability score (viab/T_viab), y = that stage's lambda, color =
    selectivity score (sel/T_sel) — the two scores are kept on separate visual
    channels instead of being summed into one axis, so both are readable per
    point. Same x-axis and color scale in every panel, so panels are directly
    comparable. Each panel's legend reports how many variants are still
    findable (non-zero) at that stage.

    protocol : a Protocol instance
    run_loop : if True, calls protocol.loop_DE() to get a fresh round; if False,
               uses whatever lambda0..lambda4/lambda0p/lambda2p/lambda3p are
               already set on protocol (e.g. after a manual step-by-step run)
    """
    if run_loop:
        protocol.loop_DE()

    viab_scores = np.array(protocol.compute_score(protocol.F_viab, protocol.J_viab))
    sel_scores  = np.array(protocol.compute_score(protocol.F_sel, protocol.J_sel))
    x           = viab_scores / protocol._T_viab
    c           = sel_scores / protocol._T_sel
    vmin, vmax  = float(c.min()), float(c.max())

    panels = [
        ('lambda0',  protocol.lambda0),
        ('lambda0p', protocol.lambda0p),
        ('lambda1',  protocol.lambda1),
        ('lambda2',  protocol.lambda2),
        ('lambda2p', protocol.lambda2p),
        ('lambda3',  protocol.lambda3),
        ('lambda3p', protocol.lambda3p),
        ('lambda4',  protocol.lambda4),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    for ax, (name, y_arr) in zip(axes, panels):
        y = np.array(y_arr)
        mask = y > 0
        n_found = int(mask.sum())
        if mask.any():
            sca = ax.scatter(x[mask], y[mask], c=c[mask], cmap='RdYlGn',
                              vmin=vmin, vmax=vmax, alpha=1, s=45,
                              edgecolors='black', linewidths=0.6,
                              label=f'{n_found} variants findable')
            ax.set_yscale('log')
            plt.colorbar(sca, ax=ax, label='Selectivity score (sel/T_sel)', fraction=0.046, pad=0.04)
            ax.legend(loc='upper left', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
        ax.set_xlabel('Viability score (viab/T_viab)')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# 10 - Viability vs selectivity score, colored by each lambda stage
def plot_all_lambda_scores(protocol, title="Viability vs selectivity score, colored by each lambda"):
    """
    2x4 grid, one panel per lambda (lambda0, lambda0p, lambda1, lambda2, lambda2p,
    lambda3, lambda3p, lambda4) -- same recipe as plot_initial_library_scores
    (x = viability score, y = selectivity score, color = abundance at that stage),
    but one panel per stage instead of only lambda0. Color uses a log scale (counts
    span many orders of magnitude), so each panel only plots sequences with nonzero
    abundance at that particular stage. Every panel shares the SAME x/y axis limits
    (the full viab_scores/sel_scores range, computed once) instead of each Axes
    auto-scaling to just its own surviving/masked points -- otherwise panels with
    fewer surviving variants (e.g. lambda3p late in the pipeline) silently zoom in,
    making the spread of points look different across panels for a reason that has
    nothing to do with the data itself.

    protocol : a Protocol instance that has already run at least one round (e.g.
               via loop_DE()), so all lambda* attributes hold real values.
    """
    viab_scores = np.array(protocol.compute_score(protocol.F_viab, protocol.J_viab))
    sel_scores  = np.array(protocol.compute_score(protocol.F_sel, protocol.J_sel))

    x_pad = 0.02 * (viab_scores.max() - viab_scores.min())
    y_pad = 0.02 * (sel_scores.max() - sel_scores.min())
    xlim  = (viab_scores.min() - x_pad, viab_scores.max() + x_pad)
    ylim  = (sel_scores.min() - y_pad, sel_scores.max() + y_pad)

    panels = [
        ('lambda0',  protocol.lambda0),
        ('lambda0p', protocol.lambda0p),
        ('lambda1',  protocol.lambda1),
        ('lambda2',  protocol.lambda2),
        ('lambda2p', protocol.lambda2p),
        ('lambda3',  protocol.lambda3),
        ('lambda3p', protocol.lambda3p),
        ('lambda4',  protocol.lambda4),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    for ax, (name, counts) in zip(axes, panels):
        counts = np.array(counts)
        mask = counts > 0
        if mask.any():
            sca = ax.scatter(viab_scores[mask], sel_scores[mask], c=counts[mask],
                              cmap='viridis', norm=LogNorm(), s=14, alpha=0.8,
                              edgecolors='none')
            fig.colorbar(sca, ax=ax, label=name, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Viability score')
        ax.set_ylabel('Selectivity score')
        ax.set_title(name)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_lambda3p_dilution_scatter(protocol, dilution_factors, title=None):
    """
    Scatter of every variant (x = viability score, y = selectivity score), layered
    once per dilution_factor: smallest dilution (most variants found) is drawn first
    in red, then each successively larger dilution_factor is drawn on top moving
    toward blue.

    Because later (bluer) layers are drawn on top wherever a variant is still found,
    a point that stays visibly RED marks a variant that was found at a weak dilution
    but LOST once dilution increased. A point that ends up BLUE survived even the
    strongest dilution tested. This makes the inclusion/attrition pattern directly
    readable off the plot: cluster the surviving (blue) points against viability/
    selectivity score to see whether loss is uniform or concentrated on
    low-score (inefficient) variants -- which is expected, since low-score variants
    sit at lower abundance and are the first to be zeroed out by the dilution's
    Poisson bottleneck.

    NOTE: this inclusion is only approximate (~97-99.8% overlap between consecutive
    levels, verified numerically), not exact -- each dilution_factor redraws
    independent randomness rather than literally sub-sampling the previous draw.

    Each dilution_factor gets its own discrete color (not a gradient) -- smallest
    dilution first (drawn at the back), largest last (drawn on top), so a point still
    showing an earlier (smaller-dilution) color was lost once dilution increased.

    Uses protocol.lambda3 as the pre-NGS pool (i.e. plots lambda3p at each dilution).
    """
    dilution_factors = sorted(float(d) for d in dilution_factors)
    viab_scores = np.array(protocol.compute_score(protocol.F_viab, protocol.J_viab))
    sel_scores  = np.array(protocol.compute_score(protocol.F_sel, protocol.J_sel))

    # Fixed, explicit palette (avoids relying on Colormap.colors, which static
    # type checkers like Pylance don't recognize on the generic Colormap type).
    _PALETTE = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(dilution_factors))]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    orig_dilution = protocol.dilution_factor
    for i, d in enumerate(dilution_factors):  # smallest first -> back layer; largest last -> front layer
        protocol.dilution_factor = d
        lam3p = np.array(protocol.NGS(protocol.lambda3))
        mask = lam3p > 0
        n_found = int(mask.sum())
        ax.scatter(viab_scores[mask], sel_scores[mask], color=colors[i],
                   s=28, edgecolors='none', alpha=0.85, zorder=i,
                   label=f'dilution_factor = {d:g}  (n={n_found})')
    protocol.dilution_factor = orig_dilution

    ax.set_xlabel('Viability score')
    ax.set_ylabel('Selectivity score')
    ax.set_title(title or 'lambda3p variants retained across dilution factors')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(title='Dilution factor', loc='best', framealpha=0.9)
    fig.tight_layout()
    return fig


# 9 - Entropy vs sequencing depth: stochastic NGS reads vs the noise-free true pool
#
# Ported from the collaborator's "model 3" notebook, section "Part VII -
# Comparison randomness vs deterministic". That notebook ran a fully separate
# noise-free ("deterministic") simulation of the whole protocol to get a
# reference entropy; this Protocol class doesn't have an equivalent no-noise
# simulation mode, so the reference used here is the entropy of the TRUE pool
# itself (i.e. the counts *before* NGS is applied) -- which is exactly what the
# noise-free simulation was converging to in the original notebook, since NGS
# noise is the only thing separating "true pool" from "sequenced reads".
def plot_entropy_vs_depth(protocol, D_values, pool='lambda0', title=None):
    """
    Sweeps the NGS sequencing depth D and plots how the Shannon entropy
    measured from the resulting reads compares to two noise-free references:
    the entropy of the TRUE underlying pool (what you'd measure with infinite
    sequencing depth) and the theoretical maximum entropy (a perfectly uniform
    pool spread over however many sequences are actually present).

    Why this curve is useful: NGS is a two-stage sampling process (pipetting,
    then PCR + sequencing) layered on top of the true pool. At low D that
    sampling is noisy and under-represents rare sequences, which distorts the
    measured entropy relative to the truth; as D grows, sampling noise shrinks
    and the measured entropy should converge onto the true pool's entropy (the
    red dashed line). A curve that DOESN'T converge as D grows a long way
    signals a bug in the NGS pipeline, not just "normal" sampling noise.

    protocol : a Protocol instance. protocol.D is temporarily overwritten for
               each value in D_values and restored to its original value once
               the sweep finishes (even if an error is raised partway through),
               so this is safe to call on a protocol you're still using
               elsewhere.
    D_values : sequencing depths to sweep, e.g. np.logspace(6, 18, 13). Plotted
               on a log x-axis (sequencing depth realistically spans many
               orders of magnitude).
    pool     : name of the protocol attribute to sequence at each depth, e.g.
               'lambda0' (initial library, default) or 'lambda3' (post-
               selectivity pool) -- whichever stage you want to probe.

    Returns
    -------
    fig, H_ngs : the matplotlib figure and the list of measured NGS-read
                 entropies (in nats, one per D_values entry, same convention as
                 shannon_entropy elsewhere in this file), in case further
                 analysis of the sweep is needed.
    """
    true_pool = np.array(getattr(protocol, pool))
    H_true    = shannon_entropy(true_pool)
    H_max     = float(np.log(max(int(np.sum(true_pool > 0)), 1)))

    D_values   = list(D_values)
    original_D = protocol.D
    H_ngs = []
    try:
        for D in D_values:
            protocol.D = float(D)
            reads = np.array(protocol.NGS(getattr(protocol, pool)))
            H_ngs.append(shannon_entropy(reads))
    finally:
        protocol.D = original_D  # always restore, even if NGS raised mid-sweep

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(D_values, H_ngs, marker='o', linestyle='-', color='tab:blue',
            label='NGS reads (stochastic)')
    ax.axhline(H_true, color='tab:red', linestyle='--', linewidth=2,
               label=f'True {pool} entropy (noise-free)')
    ax.axhline(H_max, color='black', linestyle=':', linewidth=1.5,
               label='log(support size) (uniform pool)')

    ax.set_xscale('log')
    ax.set_xlabel('Sequencing depth D')
    ax.set_ylabel('Shannon entropy (nats)')
    ax.set_title(title or f'Shannon entropy vs sequencing depth ({pool})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    fig.tight_layout()
    return fig, H_ngs


# 11 - Heatmap: top-k recovery across a 2D (N1, rho) parameter sweep
#
# Ported from the collaborator's "model 3" notebook, section "Heatmap
# analysis". The notebook swept its own library class's N1/d0 and rho
# parameters; here it's adapted to this file's Protocol class, which exposes
# the same two knobs as protocol.N1 and protocol._rho (stored with a leading
# underscore -- there's no public `rho` attribute on Protocol, so that's the
# name to pass values through when reading results back).
def plot_topk_recovery_heatmap(protocol, N1_values, rho_values, k_frac=0.10,
                                n_repeats=5, n_rounds=1, final_stage='lambda4',
                                title=None):
    """
    2D parameter-sweep heatmap of top-k recovery over (N1, rho): how well the
    sequences that end up MOST ABUNDANT after directed evolution match the
    sequences that are truly best by the underlying viability + selectivity
    score, as a function of two parameters that control how harsh the
    population bottleneck is:
      - N1  : plasmid molecules sampled/pipetted at the start of each round
              (more molecules = less chance of losing a sequence to sampling
              noise alone)
      - rho : HEK cells transfected per plasmid molecule (fewer cells per
              molecule = fewer independent chances for a sequence to be
              captured and expressed)

    How each grid cell is computed
    -------------------------------
    1. The "ground truth" top-k set is fixed ONCE, before the sweep, from
       compute_score(F_viab, J_viab)/T_viab + compute_score(F_sel, J_sel)/T_sel
       on protocol.sequence. This doesn't depend on any stochastic step, so
       reusing the same fixed set for every grid point keeps the comparison
       apples-to-apples.
    2. For every (N1, rho) grid point: protocol.N1 and protocol._rho are
       overwritten, then protocol.N_loop_DE(n_rounds) is run `n_repeats` times
       (N_loop_DE already restarts from a fresh uniform lambda0 each call, so
       every repeat is an independent trial). Each repeat's own top-k -- by
       abundance in `final_stage` -- is compared against the fixed
       ground-truth top-k, and the overlap fraction (recovered / k) is
       recorded.
    3. mean_recovery[i, j] is the average of that fraction over the n_repeats
       runs at grid point (rho_values[i], N1_values[j]); std_recovery[i, j] is
       its standard deviation, in case a cell's noise level matters to you.

    The heatmap uses log-log axes (both parameters typically span several
    decades), a viridis colormap fixed to [0, 1] (recovery is a fraction),
    white contour lines at fixed values of mu = rho * N1 / d0 (the expected
    number of HEK cells transfected per plasmid molecule, averaged across the
    library -- the single combined quantity that mostly determines how harsh
    the bottleneck is), and a red star marking the protocol's ORIGINAL (N1,
    rho) values from before the sweep.

    protocol    : a Protocol instance. protocol.N1, protocol._rho and every
                  lambda* attribute are overwritten during the sweep and
                  RESTORED to their original values once it finishes (even if
                  it raises), so this is safe to call on a protocol you're
                  still using elsewhere. This is an EXPENSIVE call -- it runs
                  len(N1_values) * len(rho_values) * n_repeats full
                  directed-evolution rounds (with a tqdm progress bar per
                  round, from N_loop_DE) -- so keep the grid small (e.g. 6-10
                  points per axis) for a first look.
    N1_values   : plasmid molecules sampled per round to sweep (x-axis), e.g.
                  np.geomspace(1e5, 1e9, 8)
    rho_values  : HEK cells transfected per plasmid to sweep (y-axis), e.g.
                  np.geomspace(1e-6, 1e-3, 8)
    k_frac      : top fraction of the library defining "top-k" (default 10%)
    n_repeats   : independent stochastic repeats averaged per grid point
    n_rounds    : directed-evolution rounds per repeat (passed to
                  N_loop_DE) -- default 1 for a lighter single-round sweep;
                  raise it to see how the bottleneck compounds over several
                  successive rounds, like the notebook's multi-cycle sweep did
    final_stage : name of the protocol attribute whose abundance ranks the
                  "recovered" top-k at each repeat (default 'lambda4', the
                  final post-amplification pool)

    Returns
    -------
    fig, mean_recovery, std_recovery : matplotlib figure and the two
    (len(rho_values), len(N1_values)) result matrices, in case further
    analysis of the sweep is needed.
    """
    N1_values  = np.asarray(list(N1_values), dtype=float)
    rho_values = np.asarray(list(rho_values), dtype=float)

    combined_gt = (np.array(protocol.compute_score(protocol.F_viab, protocol.J_viab)) / protocol._T_viab
                   + np.array(protocol.compute_score(protocol.F_sel, protocol.J_sel)) / protocol._T_sel)
    d0 = protocol.d0
    k  = max(int(d0 * k_frac), 1)
    top_gt = set(np.argsort(-combined_gt)[:k])

    original_N1, original_rho = protocol.N1, protocol._rho
    original_lambdas = {name: getattr(protocol, name) for name in
                         ('lambda0', 'lambda0p', 'lambda1', 'lambda2', 'lambda2p',
                          'lambda3', 'lambda3p', 'lambda4')}

    mean_recovery = np.zeros((len(rho_values), len(N1_values)))
    std_recovery  = np.zeros_like(mean_recovery)

    try:
        for i, rho_val in enumerate(rho_values):
            for j, N1_val in enumerate(N1_values):
                protocol.N1   = float(N1_val)
                protocol._rho = float(rho_val)

                recoveries = []
                for _ in range(n_repeats):
                    protocol.N_loop_DE(n_rounds)
                    final_pool = np.array(getattr(protocol, final_stage))
                    present    = np.flatnonzero(final_pool > 0)
                    if len(present) == 0:
                        recoveries.append(0.0)
                        continue
                    top_hat = set(present[np.argsort(-final_pool[present])[:k]])
                    recoveries.append(len(top_gt & top_hat) / k)

                mean_recovery[i, j] = np.mean(recoveries)
                std_recovery[i, j]  = np.std(recoveries) if n_repeats > 1 else 0.0
    finally:
        protocol.N1, protocol._rho = original_N1, original_rho
        for name, value in original_lambdas.items():
            setattr(protocol, name, value)

    X, Y    = np.meshgrid(N1_values, rho_values)
    mu_grid = X * Y / d0

    fig, ax = plt.subplots(figsize=(9, 6.5))
    hm = ax.pcolormesh(N1_values, rho_values, mean_recovery, shading='nearest',
                        cmap='viridis', vmin=0, vmax=1)
    ax.set_xscale('log')
    ax.set_yscale('log')

    requested_mu_levels = [0.1, 1, 3, 5, 10, 30, 100, 300, 1000]
    mu_levels = [m for m in requested_mu_levels if mu_grid.min() <= m <= mu_grid.max()]
    if mu_levels:
        cs = ax.contour(X, Y, mu_grid, levels=mu_levels, colors='white', linewidths=1.2)
        ax.clabel(cs, fmt={m: rf'$\mu={m:g}$' for m in mu_levels}, fontsize=8)

    current_mu = original_N1 * original_rho / d0
    ax.scatter(original_N1, original_rho, marker='*', s=200, color='red',
               edgecolor='white', linewidth=1, zorder=5,
               label=f'Current point (mu={current_mu:.2f})')

    fig.colorbar(hm, ax=ax, label=f'Mean top-{k} recovery ({n_repeats} repeats)')
    ax.set_xlabel('N1 (plasmid molecules sampled)')
    ax.set_ylabel(r'$\rho$ (HEK cells per plasmid)')
    ax.set_title(title or f'Top-{k} recovery across (N1, rho) -- {final_stage}')
    ax.legend(loc='lower left')
    fig.tight_layout()
    return fig, mean_recovery, std_recovery

#######################################################################################
#######################################################################################
#######################################################################################