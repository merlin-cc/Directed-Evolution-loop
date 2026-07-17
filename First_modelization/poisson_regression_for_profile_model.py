# %% [markdown]
# # Poisson Regression for the AAV Linear Model
# 
# This notebook implements Poisson GLM as an alternative to Ridge regression on log-ratios.
# It follows the two-step protocol:
# 1. **Viability** : $n^{(0)} \to \lambda_1'$ — Poisson GLM with offset $\log n_n^{(0)}$
# 2. **Selectivity** : $\lambda_1' \to n^{(2)}$ — Poisson GLM with offset $\log \lambda_{1,n}'$
# 
# See `Poisson_regression_AAV.tex` for the mathematical justification.

# %% [markdown]
# ## 1. Imports & shared setup
# 
# Reuses `sequences`, `n0`, `lambda1p`, `n2`, `X_bias`, `T_viab`, `T_sel` from `first_ML_model.ipynb`.
# Run that notebook first, or regenerate the data below.

# %%
import numpy as np
import jax
import jax.numpy as jnp
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sequence_classes import protocol, calculate_scores, T_sel, T_viab
from analysis import pearson, precision_at_k

key = jax.random.key(42)
key_seq, key_viab_w, key_sel_w, key_n0, key_run = jax.random.split(key, 5)

num_amino_acids = 20
num_positions   = 7
num_sequences   = 1_000_000

sequences = jax.random.randint(
    key_seq, shape=(num_sequences, num_positions),
    minval=0, maxval=num_amino_acids
)

n0 = jax.random.randint(
    key_n0, shape=(num_sequences,), minval=10, maxval=1000
).astype(jnp.float32)

viability_weights_GT   = jax.random.normal(key_viab_w, shape=(num_amino_acids, num_positions))
selectivity_weights_GT = jax.random.normal(key_sel_w,  shape=(num_amino_acids, num_positions))

_, capsids, lambda1p, n2 = protocol(
    n0, sequences,
    viability_weights_GT,
    selectivity_weights_GT,
    use_full_ngs=False
)

X_oh   = jax.nn.one_hot(sequences, num_amino_acids)
X_flat = np.array(X_oh.reshape(num_sequences, -1))
X_bias = np.hstack([X_flat, np.ones((num_sequences, 1))])

# %%
# ── Optimal α values (update after running Section 6, then skip the sweep) ──
alpha_opt_viab = 1e-8
alpha_opt_sel  = 1e-8


# %% [markdown]
# ## 2. Shared utilities

# %%
def extract_weights(glm_result, T):
    """
    Reshape the GLM coefficient vector back to a (20, 7) weight matrix.
    Multiply by T to recover the scale of the original score function.
    """
    coef = glm_result.params
    w = coef[:-1].reshape(num_positions, num_amino_acids).T
    return jnp.array(w * T)

# %% [markdown]
# ## 3. Viability — Poisson GLM
# 
# ### Model
# 
# $$
# \lambda_{1,n}' \sim \mathrm{Poisson}\!\left(\exp\!\left(\log n_n^{(0)} + \frac{\mathbf{x}_n^\top \mathbf{w}^\mathrm{viab}}{T_\mathrm{viab}} + c\right)\right)
# $$
# 
# The offset $\log n_n^{(0)}$ is known and fixed. `statsmodels` accepts it via the `offset` argument of `GLM`.
# 
# Zero-count sequences ($\lambda_{1,n}' = 0$) are **included** — they contribute $-\mu_n$ to the log-likelihood and push the predicted rate down for low-viability variants.  
# No masking or pseudocount is needed.
# 

# %%
offset_viab = np.log(np.array(n0, dtype=float) + 0.5)

result_viab = sm.GLM(
    endog  = np.array(lambda1p, dtype=float),
    exog   = X_bias,
    family = sm.families.Poisson(),
    offset = offset_viab,
).fit_regularized(alpha=1e-8, L1_wt=0.0, maxiter=50)

print(f'Params shape : {result_viab.params.shape}')
print(f'Params NaN   : {np.isnan(result_viab.params).sum()}')
print(f'|params| max : {np.abs(result_viab.params).max():.4f}')


# %%
viability_weights_poisson = extract_weights(result_viab, T_viab)

v_gt          = calculate_scores(sequences, viability_weights_GT)
v_hat_poisson = calculate_scores(sequences, viability_weights_poisson)

print(f'Pearson r (scores) : {pearson(v_gt, v_hat_poisson):.4f}')
print(f'Pearson r (weights): {pearson(np.array(viability_weights_poisson).flatten(), np.array(viability_weights_GT).flatten()):.4f}')

# %% [markdown]
# ### Résultats du GLM (α≈0) — lecture
# 
# **Pourquoi `fit_regularized(alpha=1e-8)` et pas `fit()` ?**  
# Le one-hot a 7 dépendances structurelles : à chaque position $l$, les 20 colonnes indicatrices somment au vecteur biais.  
# `fit()` détecte ce rang déficient et met les 7 paramètres redondants à `nan`, ce qui propage des `nan` dans les scores prédits.  
# Un $\alpha=10^{-8}$ rend $X^\top X + \alpha I$ plein-rang sans changer le fit de façon visible.
# 
# **Pearson r (scores)** — corrélation entre $\hat{f}_n$ et $f_n^{\mathrm{GT}}$ sur les $N$ séquences.  
# C'est la métrique utile : un $r$ élevé signifie qu'on classe les variants dans le bon ordre.
# 
# **Pearson r (weights)** — compare les 140 entrées de $\hat{\mathbf{w}}$ aux GT directement.  
# Plus pessimiste : les poids ne sont identifiables qu'à une constante additive par position (la colonne biais absorbe la moyenne).
# 
# **Ce qu'on attend ici :**  
# Le fit inclut les 95 % de séquences avec $\lambda_{1,n}'=0$ — un signal diffus mais réel.  
# Avec $N=10^6$ et $D=50\,000$, le rapport signal/bruit est très faible ; on s'attend à un $r$ (scores) modeste.  
# `std(hat)` ≪ `std(GT)` confirmera que les poids sont écrasés vers zéro.  
# La Section 4 (CV + $\alpha$ optimal) cherche à récupérer plus de signal.
# 

# %% [markdown]
# ## 4. Regularised Poisson Viability (L2 penalty)
# 
# `sm.GLM.fit_regularized` adds an $\ell_2$ penalty $\lambda \|\mathbf{w}\|^2$ to the Poisson log-likelihood.  
# The IRLS update becomes $(X^\top W X + \lambda I)^{-1} X^\top W z$ — same structure as Ridge.  
# Select $\lambda$ by 5-fold cross-validation on the **Poisson deviance** rather than MSE.
# 
# $$
# D(y, \hat{\mu}) = 2 \sum_n \left[ y_n \log\frac{y_n}{\hat{\mu}_n} - (y_n - \hat{\mu}_n) \right]
# $$
# 

# %%
def poisson_deviance(y, mu_hat):
    """
    Mean per-sample Poisson deviance
    """
    log_term = np.where(y > 0, y * np.log(np.maximum(y, 1e-10) / np.maximum(mu_hat, 1e-10)), 0.0)
    return 2 * np.mean(log_term - (y - mu_hat))


def poisson_cv_deviance(X, y, offset, alphas, kf):
    """
    K-fold CV on Poisson deviance for a range of L2 penalties.
    
    Parameters
    ----------
    X      : (M, p) design matrix  — masked to lambda1p > 0
    y      : (M,)   response counts
    offset : (M,)   log-offset
    alphas : array of L2 penalty strengths
    kf     : KFold splitter
    """
    deviances = np.zeros(len(alphas))

    for i, alpha in enumerate(alphas):
        fold_devs = []
        for tr, val in kf.split(X):
            res = sm.GLM(
                endog  = y[tr],
                exog   = X[tr],
                family = sm.families.Poisson(),
                offset = offset[tr],
            ).fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=30)

            mu_val = res.predict(exog=X[val], offset=offset[val])
            fold_devs.append(poisson_deviance(y[val], mu_val))

        deviances[i] = np.mean(fold_devs)
        print(f'  α={alpha:8.6f}  deviance={deviances[i]:.6f}', flush=True)

    return deviances



mask_viab  = np.array(lambda1p > 0)
X_viab_m   = X_bias[mask_viab]
y_viab_m   = np.array(lambda1p, dtype=float)[mask_viab]
off_viab_m = offset_viab[mask_viab]

print(f'CV sur {mask_viab.sum()} séquences informatives')

alphas  = np.logspace(-9, 2, 100)
K_FOLDS = 5
kf      = KFold(n_splits=K_FOLDS, shuffle=True, random_state=0)

cv_deviance_viab = poisson_cv_deviance(X_viab_m, y_viab_m, off_viab_m, alphas, kf)

alpha_best_viab = alphas[np.argmin(cv_deviance_viab)]
print(f'\nBest α viabilité : {alpha_best_viab:.4f}')

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(alphas, cv_deviance_viab, color='steelblue', lw=2, marker='o', ms=5)
ax.axvline(alpha_best_viab, color='k', linestyle='--', lw=1,
           label=f'best α = {alpha_best_viab:.4f}')
ax.set_xlabel('α (log scale)', fontsize=14)
ax.set_ylabel(f'{K_FOLDS}-fold Poisson deviance', fontsize=14)
ax.set_title('Viabilité — CV deviance vs α')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11, borderaxespad=0.3)
plt.tight_layout(); plt.show()


# %%
result_viab_reg = sm.GLM(
    endog  = np.array(lambda1p, dtype=float),
    exog   = X_bias,
    family = sm.families.Poisson(),
    offset = offset_viab,
).fit_regularized(alpha=alpha_best_viab, L1_wt=0.0, maxiter=50)

viability_weights_poisson = extract_weights(result_viab_reg, T_viab)
v_hat_poisson = calculate_scores(sequences, viability_weights_poisson)

print(f'Pearson r (scores)  : {pearson(v_gt, v_hat_poisson):.4f}')
print(f'Pearson r (weights) : {pearson(np.array(viability_weights_poisson).flatten(), np.array(viability_weights_GT).flatten()):.4f}')


# %% [markdown]
# ## 5. Selectivity — Poisson GLM
# 
# ### Standard approach (noisy offset)
# 
# Treat the observed $\lambda_{1,n}'$ as a fixed offset:
# $$
# n_n^{(2)} \sim \mathrm{Poisson}\!\left(\exp\!\left(\log \lambda_{1,n}' + \frac{\mathbf{x}_n^\top \mathbf{w}^\mathrm{sel}}{T_\mathrm{sel}} + c\right)\right)
# $$
# 
# **Caveat**: sequences with $\lambda_{1,n}' = 0$ cannot be included (offset = $-\infty$). They must be masked — same as Ridge.  
# The errors-in-variables bias is largest for sequences with few viability reads.
# 

# %%
mask_sel   = np.array(lambda1p > 0)
X_sel      = X_bias[mask_sel]
n2_sel     = np.array(n2, dtype=float)[mask_sel]
offset_sel = np.log(np.array(lambda1p, dtype=float)[mask_sel])

print(f'CV sur {mask_sel.sum()} séquences informatives (sélectivité)')

# Cross Validation
cv_deviance_sel = poisson_cv_deviance(X_sel, n2_sel, offset_sel, alphas, kf)
alpha_best_sel  = alphas[np.argmin(cv_deviance_sel)]
print(f'\nBest α sélectivité : {alpha_best_sel:.4f}')

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(alphas, cv_deviance_sel, color='darkorange', lw=2, marker='o', ms=5)
ax.axvline(alpha_best_sel, color='k', linestyle='--', lw=1, label=f'best α = {alpha_best_sel:.4f}')
ax.set_xlabel('α (log scale)', fontsize=14)
ax.set_ylabel(f'{K_FOLDS}-fold Poisson deviance', fontsize=14)
ax.set_title('Sélectivité (offset bruité) — CV deviance vs α')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11, borderaxespad=0.3)
plt.tight_layout(); plt.show()

result_sel = sm.GLM(
    endog  = n2_sel,
    exog   = X_sel,
    family = sm.families.Poisson(),
    offset = offset_sel,
).fit_regularized(alpha=alpha_best_sel, L1_wt=0.0, maxiter=50)

selectivity_weights_poisson = extract_weights(result_sel, T_sel)
s_gt          = calculate_scores(sequences, selectivity_weights_GT)
s_hat_noisy   = calculate_scores(sequences, selectivity_weights_poisson)

print(f'Pearson r (scores)  : {pearson(s_gt, s_hat_noisy):.4f}')
print(f'Pearson r (weights) : {pearson(np.array(selectivity_weights_poisson).flatten(), np.array(selectivity_weights_GT).flatten()):.4f}')


# %%
fig, axes = plt.subplots(3, 2, figsize=(12, 13))

# ── Rows 0-1: Poisson | GT weight matrices ───────────────────────────────
for row, (w_gt_m, w_hat, label, a_row) in enumerate([
    (viability_weights_GT,   viability_weights_poisson, 'Viability',   alpha_best_viab),
    (selectivity_weights_GT, selectivity_weights_poisson, 'Selectivity', alpha_best_sel),
]):
    vmax = float(jnp.abs(jnp.stack([w_gt_m, w_hat])).max())
    for col, (data, title) in enumerate([
        (w_hat,  f'{label} — Poisson (α={a_row:.2e})'),
        (w_gt_m, f'{label} — GT'),
    ]):
        ax = axes[row, col]
        im = ax.imshow(np.array(data), cmap='RdBu', aspect='auto',
                       vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Position', fontsize=12); ax.set_ylabel('AA index', fontsize=12)
        ax.set_xticks(range(num_positions))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# ── Row 2: score scatter Poisson vs GT ────────────────────────────────────
for col, (gt, w_hat, label, color, a_row) in enumerate([
    (v_gt, viability_weights_poisson,   'Viability',   'steelblue',  alpha_best_viab),
    (s_gt, selectivity_weights_poisson, 'Selectivity', 'darkorange', alpha_best_sel),
]):
    ax = axes[2, col]
    hat = np.array(calculate_scores(sequences, w_hat))
    ax.scatter(gt, hat, alpha=0.3, s=5, color=color)
    lo = float(min(np.array(gt).min(), hat.min()))
    hi = float(max(np.array(gt).max(), hat.max()))
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    r_val = pearson(gt, hat)
    ax.set_title(f'{label} — Poisson α={a_row:.2e}   r={r_val:.3f}', fontsize=9)
    ax.set_xlabel('GT score', fontsize=14); ax.set_ylabel('Predicted score', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)

plt.suptitle(
    f'Poisson GLM (α_CV) — weight recovery & score prediction',
    fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Effect of regularisation strength — why α≈0 wins
# 
# The CV in Section 4 uses a coarse grid `logspace(-2, 2)`.
# The best deviance may fall at the leftmost point, suggesting the optimum is at **very low α**.
# 
# This section sweeps a fine grid over `[1e-8, 1e-1]` and plots:
# - Pearson r (scores) vs α — shows the signal captured
# - Poisson deviance vs α — the CV criterion
# 
# The hypothesis: the Poisson likelihood itself provides implicit regularisation through count
# structure. A small α is needed only to resolve the rank-deficiency (7 structural dependencies
# of the one-hot encoding); beyond that, more regularisation hurts.
# 

# %%
# After some computation, low values for alpha performed the best
alphas_fine = np.logspace(-8, 0, 5)

pearson_viab_vs_alpha = []
pearson_sel_vs_alpha  = []

for alpha in alphas_fine:
    # Viability
    res_v = sm.GLM(
        endog=np.array(lambda1p, dtype=float), exog=X_bias,
        family=sm.families.Poisson(), offset=offset_viab,
    ).fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=50)
    wv = extract_weights(res_v, T_viab)
    vh = calculate_scores(sequences, wv)
    pearson_viab_vs_alpha.append(pearson(v_gt, vh))

    # Selectivity (noisy offset, masked)
    res_s = sm.GLM(
        endog=n2_sel, exog=X_sel,
        family=sm.families.Poisson(), offset=offset_sel,
    ).fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=50)
    ws = extract_weights(res_s, T_sel)
    sh = calculate_scores(sequences, ws)
    pearson_sel_vs_alpha.append(pearson(s_gt, sh))

    print(f'α={alpha:.2e}  r_viab={pearson_viab_vs_alpha[-1]:+.4f}  r_sel={pearson_sel_vs_alpha[-1]:+.4f}',
          flush=True)

pearson_viab_vs_alpha = np.array(pearson_viab_vs_alpha)
pearson_sel_vs_alpha  = np.array(pearson_sel_vs_alpha)


# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

for ax, r_vals, label, color in [
    (axes[0], pearson_viab_vs_alpha, 'Viabilité',   'steelblue'),
    (axes[1], pearson_sel_vs_alpha,  'Sélectivité', 'darkorange'),
]:
    ax.semilogx(alphas_fine, r_vals, color=color, lw=2, marker='o', ms=5)
    best_idx = np.argmax(r_vals)
    ax.axvline(alphas_fine[best_idx], color='k', linestyle='--', lw=1,
               label=f'best α = {alphas_fine[best_idx]:.2e}  (r={r_vals[best_idx]:.4f})')
    ax.set_xlabel('α (log scale)')
    ax.set_ylabel('Pearson r (scores vs GT)')
    ax.set_title(f'{label} — Pearson r vs α')
    ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.4)

plt.suptitle('Poisson GLM — signal recovered as function of L2 penalty', fontsize=12)
plt.tight_layout(); plt.show()

# Best alphas from signal perspective
alpha_opt_viab = alphas_fine[np.argmax(pearson_viab_vs_alpha)]
alpha_opt_sel  = alphas_fine[np.argmax(pearson_sel_vs_alpha)]
print(f'α optimal (Pearson) — viabilité   : {alpha_opt_viab:.2e}')
print(f'α optimal (Pearson) — sélectivité : {alpha_opt_sel:.2e}')


# %%
result_viab_opt = sm.GLM(
    endog=np.array(lambda1p, dtype=float), exog=X_bias,
    family=sm.families.Poisson(), offset=offset_viab,
).fit_regularized(alpha=alpha_opt_viab, L1_wt=0.0, maxiter=50)

result_sel_opt = sm.GLM(
    endog=n2_sel, exog=X_sel,
    family=sm.families.Poisson(), offset=offset_sel,
).fit_regularized(alpha=alpha_opt_sel, L1_wt=0.0, maxiter=50)

viability_weights_opt   = extract_weights(result_viab_opt, T_viab)
selectivity_weights_opt = extract_weights(result_sel_opt,  T_sel)

v_hat_opt = calculate_scores(sequences, viability_weights_opt)
s_hat_opt = calculate_scores(sequences, selectivity_weights_opt)

print(f'Pearson r (scores)  viabilité   : {pearson(v_gt, v_hat_opt):.4f}')
print(f'Pearson r (scores)  sélectivité : {pearson(s_gt, s_hat_opt):.4f}')
print(f'Pearson r (weights) viabilité   : {pearson(np.array(viability_weights_opt).flatten(), np.array(viability_weights_GT).flatten()):.4f}')
print(f'Pearson r (weights) sélectivité : {pearson(np.array(selectivity_weights_opt).flatten(), np.array(selectivity_weights_GT).flatten()):.4f}')


# %% [markdown]
# ## 7. Precision@k

# %%
s_gt = calculate_scores(sequences, selectivity_weights_GT)

combined_gt  = np.array(v_gt)     / T_viab + np.array(s_gt)     / T_sel
combined_opt = np.array(v_hat_opt) / T_viab + np.array(s_hat_opt) / T_sel

fracs = [0.01, 0.05, 0.10, 0.20]
print('  Top-%    Precision@k   (random baseline = top-%)')
print('  ──────────────────────────────────────────────────')
for frac in fracs:
    p = precision_at_k(combined_gt, combined_opt, k_frac=frac)
    print(f'  Top {int(frac*100):3d}%      {p:.3f}          (baseline: {frac:.3f})')


# %%
k10      = int(num_sequences * 0.10)
top_gt   = set(np.argsort(-combined_gt)[:k10])
top_hat  = set(np.argsort(-combined_opt)[:k10])
both     = np.array(sorted(top_gt & top_hat))
only_gt  = np.array(sorted(top_gt  - top_hat))
only_hat = np.array(sorted(top_hat - top_gt))
rest     = np.array(sorted(set(range(num_sequences)) - top_gt - top_hat))

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(combined_gt[rest],     combined_opt[rest],     s=4,  alpha=0.2, color='lightgray')
ax.scatter(combined_gt[only_hat], combined_opt[only_hat], s=12, alpha=0.6, color='darkorange',
           label=f'False pos. ({len(only_hat)})')
ax.scatter(combined_gt[only_gt],  combined_opt[only_gt],  s=12, alpha=0.6, color='steelblue',
           label=f'Missed ({len(only_gt)})')
ax.scatter(combined_gt[both],     combined_opt[both],     s=12, alpha=0.8, color='forestgreen',
           label=f'Recovered ({len(both)})')
lo, hi = combined_gt.min(), combined_gt.max()
ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
ax.axvline(np.sort(combined_gt)[-k10],  color='steelblue',  linestyle=':', lw=1.2)
ax.axhline(np.sort(combined_opt)[-k10], color='darkorange', linestyle=':', lw=1.2)
ax.set_xlabel('GT combined score', fontsize=14)
ax.set_ylabel('Predicted combined score (Poisson)', fontsize=14)
p10 = precision_at_k(combined_gt, combined_opt, k_frac=0.10)
ax.set_title(f'Poisson GLM — Top-10% recovery   Precision = {p10:.3f}')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=12)
ax.legend(fontsize=10, markerscale=2, borderaxespad=0.3, borderpad=0.4, handlelength=1.2)
plt.tight_layout(); plt.show()


# %% [markdown]
# ## 8. Ridge vs Poisson — comparaison finale

# %%
def fit_ridge_local(X, y, T, alpha):
    n_params = X.shape[1] - 1
    P = np.zeros((X.shape[1], X.shape[1]))
    P[:n_params, :n_params] = alpha * np.eye(n_params)
    coef = np.linalg.solve(X.T @ X + P, X.T @ np.array(y))
    w = coef[:-1].reshape(num_positions, num_amino_acids).T
    return jnp.array(w * T)

def ridge_cv_mse_local(X, y, alphas, kf):
    mse = np.zeros(len(alphas))
    n_params = X.shape[1] - 1
    for i, a in enumerate(alphas):
        P = np.zeros((X.shape[1], X.shape[1]))
        P[:n_params, :n_params] = a * np.eye(n_params)
        fold_mse = []
        for tr, val in kf.split(X):
            coef = np.linalg.solve(X[tr].T @ X[tr] + P, X[tr].T @ y[tr])
            fold_mse.append(np.mean((X[val] @ coef - y[val]) ** 2))
        mse[i] = np.mean(fold_mse)
    return mse

eps       = 0.5
n0_norm   = n0       / jnp.sum(n0)       * jnp.sum(lambda1p)
l1p_norm  = lambda1p / jnp.sum(lambda1p) * jnp.sum(n2)
y_viab_r  = np.array(jnp.log((lambda1p + eps) / (n0_norm  + eps)))
y_sel_r   = np.array(jnp.log((n2       + eps) / (l1p_norm + eps)))

mask_v  = np.array(lambda1p > 0)
mask_s  = np.array(n2 > 0)
_kf     = KFold(n_splits=5, shuffle=True, random_state=0)
_alphas = np.logspace(-2, 2, 20)

cv_v = ridge_cv_mse_local(X_bias[mask_v], y_viab_r[mask_v], _alphas, _kf)
cv_s = ridge_cv_mse_local(X_bias[mask_s], y_sel_r[mask_s],  _alphas, _kf)

wv_ridge = fit_ridge_local(X_bias[mask_v], y_viab_r[mask_v], T_viab, _alphas[np.argmin(cv_v)])
ws_ridge = fit_ridge_local(X_bias[mask_s], y_sel_r[mask_s],  T_sel,  _alphas[np.argmin(cv_s)])
vh_ridge = calculate_scores(sequences, wv_ridge)
sh_ridge = calculate_scores(sequences, ws_ridge)

cgt = np.array(v_gt) / T_viab + np.array(s_gt) / T_sel

methods = {
    'Ridge (CV)':         (vh_ridge,   sh_ridge),
    'Poisson (α_CV)':     (v_hat_poisson, s_hat_noisy),
    'Poisson (α_opt)':    (v_hat_opt,  s_hat_opt),
}

print(f'  {"Méthode":<22}  r_viab   r_sel    Prec@1%  Prec@10%')
print(f'  {"─"*58}')
for name, (vh, sh) in methods.items():
    comb = np.array(vh) / T_viab + np.array(sh) / T_sel
    print(f'  {name:<22}  {pearson(v_gt,vh):+.4f}  {pearson(s_gt,sh):+.4f}  '
          f' {precision_at_k(cgt, comb, 0.01):.3f}    {precision_at_k(cgt, comb, 0.10):.3f}')


# %% [markdown]
# ## 9. Impact of noise on results.
# 
# In the method protocol, two paramaters impact on noise, the epsilon for variability and selectivity, previously, they were set to 0.5 arbitrary, we want to know if a very noisy model will affect the results significantly and especially which model between Poisson and Ridge regression is the most robust.

# %% [markdown]
# ### Which phase is more affected by noise?

# %%
# 3 epsilon values for viability noise (selectivity noise fixed at default)
epsilons_viab = [0.1, 0.3, 0.8]
eps_sel_fixed = 0.3  # keep selectivity noise at default

v_gt = calculate_scores(sequences, viability_weights_GT)
s_gt = calculate_scores(sequences, selectivity_weights_GT)

prec10_viab = []
pearson_viab_eps = []

for eps in epsilons_viab:
    _, _caps, lambda1p_eps, _n2 = protocol(
        n0, sequences,
        viability_weights_GT, selectivity_weights_GT,
        epsilon_viability=eps,
        epsilon_selectivity=eps_sel_fixed,
        use_full_ngs=False,
    )
    _offset = np.log(np.array(n0, dtype=float) + 0.5)
    _res = sm.GLM(
        endog=np.array(lambda1p_eps, dtype=float), exog=X_bias,
        family=sm.families.Poisson(), offset=_offset,
    ).fit_regularized(alpha=alpha_opt_viab, L1_wt=0.0, maxiter=50)
    _w  = extract_weights(_res, T_viab)
    _vh = np.array(calculate_scores(sequences, _w))
    _r  = pearson(v_gt, _vh)
    _p  = precision_at_k(v_gt, _vh, k_frac=0.10)
    prec10_viab.append(_p)
    pearson_viab_eps.append(_r)
    print(f'ε_viab={eps:.1f}  Pearson r={_r:.4f}  Precision@10%={_p:.3f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, vals, ylabel, title in [
    (axes[0], pearson_viab_eps, 'Pearson r',       'Pearson r (scores vs GT)'),
    (axes[1], prec10_viab,      'Precision@10%',   'Top-10% recovery'),
]:
    bars = ax.bar([str(e) for e in epsilons_viab], vals, color='steelblue', alpha=0.8, width=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_xlabel('ε viability (noise level)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Viabilité — {title} vs ε')
    ax.set_ylim(0, max(vals) * 1.2 + 0.05)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.suptitle('Impact of viability noise — Poisson GLM (α_opt)', fontsize=12)
plt.tight_layout(); plt.show()


# %%
# 3 epsilon values for selectivity noise (viability noise fixed at default)
epsilons_sel = [0.1, 0.3, 0.8]
eps_viab_fixed = 0.3  # keep viability noise at default

prec10_sel = []
pearson_sel_eps = []

for eps in epsilons_sel:
    _, _caps, lambda1p_eps2, n2_eps2 = protocol(
        n0, sequences,
        viability_weights_GT, selectivity_weights_GT,
        epsilon_viability=eps_viab_fixed,
        epsilon_selectivity=eps,
        use_full_ngs=False,
    )
    _mask       = np.array(lambda1p_eps2 > 0)
    _X_sel      = X_bias[_mask]
    _n2_sel     = np.array(n2_eps2[_mask], dtype=float)
    _lam1p_m    = np.array(lambda1p_eps2[_mask], dtype=float)
    _off_sel    = np.log(_lam1p_m + 0.5)
    _res_s = sm.GLM(
        endog=_n2_sel, exog=_X_sel,
        family=sm.families.Poisson(), offset=_off_sel,
    ).fit_regularized(alpha=alpha_opt_sel, L1_wt=0.0, maxiter=50)
    _ws  = extract_weights(_res_s, T_sel)
    _sh  = np.array(calculate_scores(sequences, _ws))
    _r   = pearson(s_gt, _sh)
    _p   = precision_at_k(s_gt, _sh, k_frac=0.10)
    prec10_sel.append(_p)
    pearson_sel_eps.append(_r)
    print(f'ε_sel={eps:.1f}  Pearson r={_r:.4f}  Precision@10%={_p:.3f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, vals, ylabel, title in [
    (axes[0], pearson_sel_eps, 'Pearson r',     'Pearson r (scores vs GT)'),
    (axes[1], prec10_sel,      'Precision@10%', 'Top-10% recovery'),
]:
    bars = ax.bar([str(e) for e in epsilons_sel], vals, color='darkorange', alpha=0.8, width=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_xlabel('ε selectivity (noise level)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Sélectivité — {title} vs ε')
    ax.set_ylim(0, max(vals) * 1.2 + 0.05)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.suptitle('Impact of selectivity noise — Poisson GLM (α_opt)', fontsize=12)
plt.tight_layout(); plt.show()


# %%
# Viability — top-10% recovery for 4 noise levels, CV re-run per ε (Poisson)
v_gt_e    = np.array(calculate_scores(sequences, viability_weights_GT))
s_gt_e    = np.array(calculate_scores(sequences, selectivity_weights_GT))
max_vgt_e = float(v_gt_e.max())
print(f'Maximum viability score is {max_vgt_e}')
eps_viab_     = [0.0, max_vgt_e / 100, max_vgt_e / 10, max_vgt_e / 3]
eps_sel_fix   = 0.3
_alphas_sweep = np.logspace(-6, 0, 12)

fig, axes = plt.subplots(1, 4, figsize=(24, 5))
axes = axes.flatten()

for ax, eps in zip(axes, eps_viab_):
    _, _caps, lam1p_e, _n2 = protocol(
        n0, sequences, viability_weights_GT, selectivity_weights_GT,
        epsilon_viability=eps, epsilon_selectivity=eps_sel_fix, use_full_ngs=False,
    )
    _offset  = np.log(np.array(n0, dtype=float) + 0.5)
    _mask_cv = np.array(lam1p_e > 0)
    _cv_devs = poisson_cv_deviance(
        X_bias[_mask_cv], np.array(lam1p_e[_mask_cv], dtype=float),
        _offset[_mask_cv], _alphas_sweep, kf,
    )
    _alpha_eps = _alphas_sweep[np.argmin(_cv_devs)]
    print(f'epsilon={eps:.2f}  alpha_CV={_alpha_eps:.2e}', flush=True)

    _res = sm.GLM(
        endog=np.array(lam1p_e, dtype=float), exog=X_bias,
        family=sm.families.Poisson(), offset=_offset,
    ).fit_regularized(alpha=_alpha_eps, L1_wt=0.0, maxiter=50)
    _w  = extract_weights(_res, T_viab)
    _vh = np.array(calculate_scores(sequences, _w))

    k10        = int(num_sequences * 0.10)
    top_gt_v   = set(np.argsort(-v_gt_e)[:k10])
    top_hat_v  = set(np.argsort(-_vh)[:k10])
    both_v     = np.array(sorted(top_gt_v & top_hat_v))
    only_gt_v  = np.array(sorted(top_gt_v  - top_hat_v))
    only_hat_v = np.array(sorted(top_hat_v - top_gt_v))
    rest_v     = np.array(sorted(set(range(num_sequences)) - top_gt_v - top_hat_v))

    ax.scatter(v_gt_e[rest_v],     _vh[rest_v],     s=2,  alpha=0.15, color='lightgray')
    ax.scatter(v_gt_e[only_hat_v], _vh[only_hat_v], s=10, alpha=0.6,  color='darkorange',
               label=f'False pos. ({len(only_hat_v)})')
    ax.scatter(v_gt_e[only_gt_v],  _vh[only_gt_v],  s=10, alpha=0.6,  color='steelblue',
               label=f'Missed ({len(only_gt_v)})')
    ax.scatter(v_gt_e[both_v],     _vh[both_v],     s=10, alpha=0.8,  color='forestgreen',
               label=f'Recovered ({len(both_v)})')
    lo, hi = v_gt_e.min(), v_gt_e.max()
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.axvline(np.sort(v_gt_e)[-k10], color='steelblue',  linestyle=':', lw=1)
    ax.axhline(np.sort(_vh)[-k10],    color='darkorange', linestyle=':', lw=1)
    p10 = precision_at_k(v_gt_e, _vh, k_frac=0.10)
    ax.set_xlabel('GT viability score', fontsize=14)
    ax.set_ylabel('Predicted viability score', fontsize=14)
    ax.set_title(f'epsilon={eps:.2f}  alpha_CV={_alpha_eps:.1e}  Prec@10%={p10:.3f}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10, markerscale=2, borderaxespad=0.3, borderpad=0.4, handlelength=1.2)

plt.suptitle('Viabilite — Top-10% recovery vs bruit (Poisson GLM, CV par epsilon)', fontsize=13)
plt.tight_layout(); plt.show()


# %%
# Selectivity — top-10% recovery for 4 noise levels, CV re-run per epsilon (Poisson)
max_sgt_e    = float(s_gt_e.max())
print(f'maximum value for selectivity scores is {max_sgt_e}')
eps_sel_      = [0.0, max_sgt_e / 100, max_sgt_e / 10, max_sgt_e / 3]
eps_viab_fix  = 0.3

fig, axes = plt.subplots(1, 4, figsize=(24, 5))
axes = axes.flatten()

for ax, eps in zip(axes, eps_sel_):
    _, _caps, lam1p_e2, n2_e2 = protocol(
        n0, sequences, viability_weights_GT, selectivity_weights_GT,
        epsilon_viability=eps_viab_fix, epsilon_selectivity=eps, use_full_ngs=False,
    )
    _mask    = np.array(lam1p_e2 > 0)
    _X_s     = X_bias[_mask]
    _n2_s    = np.array(n2_e2[_mask],    dtype=float)
    _lam_m   = np.array(lam1p_e2[_mask], dtype=float)
    _off_s   = np.log(_lam_m + 0.5)
    _cv_devs_s   = poisson_cv_deviance(_X_s, _n2_s, _off_s, _alphas_sweep, kf)
    _alpha_eps_s = _alphas_sweep[np.argmin(_cv_devs_s)]
    print(f'epsilon={eps:.2f}  alpha_CV={_alpha_eps_s:.2e}', flush=True)

    _res_s = sm.GLM(
        endog=_n2_s, exog=_X_s,
        family=sm.families.Poisson(), offset=_off_s,
    ).fit_regularized(alpha=_alpha_eps_s, L1_wt=0.0, maxiter=50)
    _ws = extract_weights(_res_s, T_sel)
    _sh = np.array(calculate_scores(sequences, _ws))

    k10        = int(num_sequences * 0.10)
    top_gt_s   = set(np.argsort(-s_gt_e)[:k10])
    top_hat_s  = set(np.argsort(-_sh)[:k10])
    both_s     = np.array(sorted(top_gt_s & top_hat_s))
    only_gt_s  = np.array(sorted(top_gt_s  - top_hat_s))
    only_hat_s = np.array(sorted(top_hat_s - top_gt_s))
    rest_s     = np.array(sorted(set(range(num_sequences)) - top_gt_s - top_hat_s))

    ax.scatter(s_gt_e[rest_s],     _sh[rest_s],     s=2,  alpha=0.15, color='lightgray')
    ax.scatter(s_gt_e[only_hat_s], _sh[only_hat_s], s=10, alpha=0.6,  color='darkorange',
               label=f'False pos. ({len(only_hat_s)})')
    ax.scatter(s_gt_e[only_gt_s],  _sh[only_gt_s],  s=10, alpha=0.6,  color='steelblue',
               label=f'Missed ({len(only_gt_s)})')
    ax.scatter(s_gt_e[both_s],     _sh[both_s],     s=10, alpha=0.8,  color='forestgreen',
               label=f'Recovered ({len(both_s)})')
    lo, hi = s_gt_e.min(), s_gt_e.max()
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.axvline(np.sort(s_gt_e)[-k10], color='steelblue',  linestyle=':', lw=1)
    ax.axhline(np.sort(_sh)[-k10],    color='darkorange', linestyle=':', lw=1)
    p10 = precision_at_k(s_gt_e, _sh, k_frac=0.10)
    ax.set_xlabel('GT selectivity score', fontsize=14)
    ax.set_ylabel('Predicted selectivity score', fontsize=14)
    ax.set_title(f'epsilon={eps:.2f}  alpha_CV={_alpha_eps_s:.1e}  Prec@10%={p10:.3f}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10, markerscale=2, borderaxespad=0.3, borderpad=0.4, handlelength=1.2)

plt.suptitle('Selectivite — Top-10% recovery vs bruit (Poisson GLM, CV par epsilon)', fontsize=13)
plt.tight_layout(); plt.show()



