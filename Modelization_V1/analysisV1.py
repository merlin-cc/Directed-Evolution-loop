import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from sequence_classesV1 import *


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

    if J is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(F, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlabel('Position')
        ax.set_ylabel('Amino acid')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label='Weight')
        fig.tight_layout()
        return fig

    J = np.array(J)
    j_strength = np.sqrt(np.sum(J ** 2, axis=(2, 3)))  # (L, L) coupling strength per position pair

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(F, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Amino acid')
    axes[0].set_title('F (profile)')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(j_strength, cmap='viridis')
    axes[1].set_xlabel('Position j')
    axes[1].set_ylabel('Position i')
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
    protocol : a Protocol instance that has already run at least sampling(),
               produce_capsids() and selectivty() (e.g. via loop_DE()), so that
               lambda0, lambda2 and lambda3 hold real values.
    Sequences are sorted by initial abundance (lambda0) for readability.
    """
    order = np.argsort(-np.array(protocol.lambda0))
    l0 = np.array(protocol.lambda0)[order]
    l2 = np.array(protocol.lambda2)[order]
    l3 = np.array(protocol.lambda3)[order]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(l0))
    ax.plot(x, l0, label='lambda0 (initial)', alpha=0.8)
    ax.plot(x, l2, label='lambda2 (post-production)', alpha=0.8)
    ax.plot(x, l3, label='lambda3 (post-selectivity)', alpha=0.8)
    ax.set_xlabel('Sequence (sorted by initial abundance)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# 4 - The constructed library for next loop, with parameter N (number of loop)
def plot_library_evolution(protocol, N, threshold=1e3):
    """
    Runs protocol.N_loop_DE(N) and plots Shannon entropy + sequence diversity of
    lambda0 (the round's starting library) across N directed-evolution rounds.

    Returns
    -------
    fig, rounds : the matplotlib figure and the raw list returned by N_loop_DE(N),
                   in case further analysis of the run is needed.
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(F_teacher, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Teacher (ground truth)')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Amino acid')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(F_student, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Student (recovered)')
    axes[1].set_xlabel('Position')
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
def plot_topk_recovery(score_gt, score_hat, k_fracs=None, title="Top-k% recovery"):
    """
    score_gt, score_hat : (num_sequences,) ground-truth and predicted scores
    k_fracs             : fractions of the library to evaluate precision_at_k at
                          (defaults to 2%-50% in 20 steps)
    """
    if k_fracs is None:
        k_fracs = np.linspace(0.02, 0.5, 20)
    precisions = [precision_at_k(score_gt, score_hat, k) for k in k_fracs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.array(k_fracs) * 100, precisions, marker='o', color='tab:blue', label='Precision@k')
    ax.plot([0, 100], [0, 1], 'k--', alpha=0.3, label='random baseline (E[precision]=k%)')
    ax.set_xlabel('k (%)')
    ax.set_ylabel('Precision@k')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    return fig

# 8 - Plotting all lambdas for a loop
def plot_loop_lambdas(protocol, run_loop=True, title="All lambdas for one DE round"):
    """
    Plots every lambda produced by a single loop_DE() round: the true-pool chain
    (lambda0 -> lambda1 -> lambda2 -> lambda3 -> lambda4, solid lines) and the
    NGS-read side-channel (lambda0p, lambda2p, lambda3p, dashed lines, never fed
    back into the true chain), all sorted by lambda0 for readability.

    protocol : a Protocol instance
    run_loop : if True, calls protocol.loop_DE() to get a fresh round; if False,
               uses whatever lambda0..lambda4/lambda0p/lambda2p/lambda3p are
               already set on protocol (e.g. after a manual step-by-step run)
    """
    if run_loop:
        protocol.loop_DE()

    order = np.argsort(-np.array(protocol.lambda0))

    bio_names = ['lambda0', 'lambda1', 'lambda2', 'lambda3', 'lambda4']
    ngs_names = ['lambda0p', 'lambda2p', 'lambda3p']

    bio_colors = plt.cm.viridis(np.linspace(0, 0.85, len(bio_names)))
    ngs_colors = plt.cm.plasma(np.linspace(0, 0.85, len(ngs_names)))

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(order))

    for name, color in zip(bio_names, bio_colors):
        y = np.array(getattr(protocol, name))[order]
        ax.plot(x, y, label=name, color=color, linestyle='-', alpha=0.85)

    for name, color in zip(ngs_names, ngs_colors):
        y = np.array(getattr(protocol, name))[order]
        ax.plot(x, y, label=name, color=color, linestyle='--', alpha=0.85)

    ax.set_xlabel('Sequence (sorted by lambda0)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    return fig

#######################################################################################
#######################################################################################
#######################################################################################