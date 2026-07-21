import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sequence_classesV1 import *

message = "file analysis 1.2"

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
    Runs N rounds of loop_DE() (same reseeding logic as protocol.N_loop_DE(N), but
    replicated here as an explicit loop so a tqdm progress bar can wrap it) and plots
    Shannon entropy + sequence diversity of lambda0 (the round's starting library)
    across N directed-evolution rounds.

    Returns
    -------
    fig, rounds : the matplotlib figure and the raw list of [bio_row, ngs_row] pairs
                   (one per round, same format as N_loop_DE's return value), in case
                   further analysis of the run is needed.
    """
    protocol.lambda0 = jnp.full(protocol.d0, protocol.N0 / protocol.d0)
    rounds = []
    for _ in tqdm(range(N), desc="Directed evolution rounds"):
        rounds.append(protocol.loop_DE())
        protocol.lambda0 = protocol.lambda4

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


# 7 - Top k% recovery
def plot_topk_recovery(score_gt, score_hat, k_frac=0.10, title=None):
    """
    GT-vs-predicted score scatter, colored by top-k% recovery category — matches
    the top-k recovery plot from Ridge_regression_for_linear_model.ipynb (cell 21):
    Recovered (top-k in both), Missed (GT top-k only), False positive
    (predicted top-k only), Rest — with the GT/predicted top-k thresholds drawn in.

    score_gt, score_hat : (num_sequences,) ground-truth and predicted scores
                          (e.g. combined = viab/T_viab + sel/T_sel)
    k_frac              : top fraction of the library defining "top-k" (default 10%)
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
    ax.set_xlabel('GT score')
    ax.set_ylabel('Predicted score')
    ax.set_title(title or f'Top-{int(k_frac * 100)}% recovery   Precision = {p:.3f}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, markerscale=2, borderaxespad=0.3, borderpad=0.4, handlelength=1.2)
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


#######################################################################################
#######################################################################################
#######################################################################################