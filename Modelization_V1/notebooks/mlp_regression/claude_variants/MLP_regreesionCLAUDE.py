#####################################################################################
#####################################################################################
#####################################################################################
#                                                                                   #
#   This file defined an MLP that will try to learn the weights in profile and      #
#   Pott's model, this consideration is due to a difficulty in finding a good       #
#   penalization factor during ridge regression, the too many weights create a      #
#   situation where the best lambda is extremely high, and therefore no choices     #
#   are made and the predicted weights are too near the mean, therefore zero        #
#                                                                                   #
#   V1 scope: PROFILE model only (F), single-site one-hot features -- no pairwise   #
#   (J) terms yet. The Potts (F + J) version is a later step (see README.md).       #
#                                                                                   #
#   Flax/Optax port: replaces the earlier PyTorch ProfileMLP with a flax.linen      #
#   Module trained through optax (same pattern as MLP_regV1.py), so this file and   #
#   RegressionV1.py's Ridge solve share one array library (jax) instead of mixing   #
#   torch + jax in the same process. BatchNorm was dropped in the port -- flax's    #
#   nn.BatchNorm needs a second mutable ("batch_stats") collection threaded through #
#   every train/eval call, which the minimal MLP_regV1.py pattern this follows      #
#   doesn't do; dropout + AdamW weight decay are the regularizers instead.          #
#                                                                                   #
#####################################################################################
#####################################################################################
#####################################################################################

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from tqdm.auto import tqdm

from sequence_classesV1 import *
from analysisV1 import pearson, precision_at_k
from RegressionV1 import build_multi_round_dataset

message = "file MLP regression 2.0 (flax/optax)"

L = 7   # num_positions, matches Protocol.compute_score's hardcoded range(7)
A = 20  # num_amino_acids


### ---------------------------- Feature construction --------------------------- ###
#######################################################################################

def build_profile_features(seqs_obs, A=A):
    """
    Single-site one-hot only -- the "profile" (F) half of
    RegressionV1.build_potts_features. Kept as its own function (not a slice of the
    Potts matrix) so V1 never constructs the O(L^2 * A^2) pairwise block at all,
    which is the actual point of a profile-only first step.

    Parameters
    ----------
    seqs_obs : (N, L) int array/sequence -- observed sequences (can repeat across
               rounds; each row is treated as one training example)
    A        : int, alphabet size

    Returns
    -------
    X : (N, L*A) float32 numpy array, no bias column (the MLP has its own biases)
    """
    seqs_obs = np.asarray(seqs_obs)
    N, Lseq = seqs_obs.shape
    oh = np.eye(A, dtype=np.float32)[seqs_obs]  # (N, L, A)
    return oh.reshape(N, Lseq * A)               # (N, L*A)


### ---------------------------- Model --------------------------- ###
#######################################################################################

class ProfileMLP(nn.Module):
    """
    MLP over single-site one-hot features -> scalar score. Mirrors what Ridge's
    F_flat @ x does for the profile model, but replaces the closed-form L2 solve
    (which collapses toward near-zero weights once the feature count balloons, see
    file header) with a network trained by SGD.
    """
    hidden_dims: tuple = (256, 128, 64)
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool):
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


### ---------------------------- Training --------------------------- ###
#######################################################################################

def train_profile_mlp(X_train, y_train, X_val, y_val, hidden_dims=(256, 128, 64),
                       dropout=0.1, epochs=300, batch_size=4096, lr=1e-3,
                       weight_decay=1e-4, patience=20, seed=0, verbose=True):
    """
    GPU-resident training: the whole profile dataset (N x L*A floats -- a few
    hundred MB even at DE_loopV1.ipynb's largest library, 800k sequences) fits
    comfortably in a DGX Spark's unified memory, so X/y are moved to jax arrays
    ONCE and every epoch's minibatches are carved out with a freshly split PRNG
    key permutation -- no DataLoader / per-step host round-trip. `train_step` is
    jax.jit-compiled once per distinct batch shape (at most twice per call: the
    full batch_size and one shorter tail batch), not retraced every step.

    Returns
    -------
    state   : dict with the best-val-MSE params plus the architecture args needed
              to rebuild the model for predict_scores (flax has no state_dict, so
              this dict is the closest equivalent).
    history : dict with "train_loss" / "val_loss" per epoch
    """
    key = jax.random.key(seed)
    key, k_init, k_drop = jax.random.split(key, 3)

    model = ProfileMLP(hidden_dims=hidden_dims, dropout=dropout)
    X_train = jnp.asarray(X_train, dtype=jnp.float32)
    y_train = jnp.asarray(y_train, dtype=jnp.float32)
    X_val   = jnp.asarray(X_val,   dtype=jnp.float32)
    y_val   = jnp.asarray(y_val,   dtype=jnp.float32)

    params = model.init({'params': k_init, 'dropout': k_drop}, X_train[:1], training=False)['params']

    n_train         = X_train.shape[0]
    steps_per_epoch = max(n_train // batch_size, 1)
    total_steps     = steps_per_epoch * epochs
    warmup_steps    = max(total_steps // 15, 1)

    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, warmup_steps=warmup_steps, decay_steps=total_steps)
    tx       = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    opt_state = tx.init(params)

    def loss_fn(params, x, y, rng, training):
        preds = model.apply({'params': params}, x, training=training, rngs={'dropout': rng})
        return jnp.mean((preds - y) ** 2)

    @jax.jit
    def train_step(params, opt_state, xb, yb, rng):
        loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb, rng, True)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_loss(params, x, y):
        return loss_fn(params, x, y, jax.random.key(0), False)

    best_val, best_params, bad_epochs = float("inf"), params, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in tqdm(range(epochs), desc="Training ProfileMLP", disable=not verbose):
        key, k_perm = jax.random.split(key)
        perm = jax.random.permutation(k_perm, n_train)
        running, seen = 0.0, 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            key, k_step = jax.random.split(key)
            xb, yb = X_train[idx], y_train[idx]
            params, opt_state, loss = train_step(params, opt_state, xb, yb, k_step)
            running += float(loss) * idx.shape[0]
            seen    += idx.shape[0]
        train_loss = running / seen

        val_loss = float(eval_loss(params, X_val, y_val))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val, bad_epochs = val_loss, 0
            best_params = params
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val MSE={best_val:.4f})")
                break

    state = dict(params=best_params, hidden_dims=hidden_dims, dropout=dropout)
    return state, history


def predict_scores(state, X):
    """Forward pass over the full X in one shot (profile features are only L*A wide)."""
    model = ProfileMLP(hidden_dims=state["hidden_dims"], dropout=state["dropout"])
    X = jnp.asarray(X, dtype=jnp.float32)
    preds = model.apply({'params': state["params"]}, X, training=False)
    return np.array(preds)


def extract_effective_F(state, T, L=L, A=A):
    """
    Single-mutant scan from an all-zero reference sequence: for every (position,
    amino acid), predict the score of a sequence with only that position set to
    that amino acid, offset by the all-blank prediction. For a model that has
    genuinely learned an additive (profile-only) function this recovers F up to
    the same per-position additive-constant ambiguity Ridge already has (see
    RegressionV1.fit_weights_potts) -- it is NOT valid once the network starts
    picking up interactions between positions, which is exactly the warning sign
    to look for when this V1 is later compared against a Potts (F + J) model.
    """
    base    = np.zeros((1, L * A), dtype=np.float32)
    mutants = np.tile(base, (L * A, 1))
    for pos in range(L):
        for aa in range(A):
            mutants[pos * A + aa, pos * A + aa] = 1.0

    scores     = predict_scores(state, np.concatenate([base, mutants], axis=0))
    base_score = scores[0]
    F_flat     = (scores[1:] - base_score).reshape(L, A)
    return jnp.array((F_flat * T).T)  # (A, L), matches fit_weights_potts' F_hat layout


### ---------------------------- End-to-end pipeline --------------------------- ###
#######################################################################################

def recover_profile_from_NGS(protocol, n_rounds=5, val_frac=0.15, eps=0.5, seed=0,
                              mlp_kwargs=None, verbose=True):
    """
    Profile-only (F) analogue of RegressionV1.recover_weights_from_NGS: builds the
    same pooled multi-round (sequence, log-ratio) dataset from protocol's observable
    NGS reads (lambda0p/lambda2p/lambda3p -- never the ground-truth lambdas), but
    fits a ProfileMLP on single-site features instead of a Ridge solve on
    single-site + pairwise features.

    Returns
    -------
    F_viab_hat, F_sel_hat, info
        info : dict with the trained model states, loss histories, and dataset sizes
    """
    mlp_kwargs = mlp_kwargs or {}

    if verbose:
        print(f"JAX backend: {jax.default_backend()} -- devices: {jax.devices()}")

    seqs_viab, y_viab, seqs_sel, y_sel = build_multi_round_dataset(protocol, n_rounds, eps=eps)

    rng = np.random.default_rng(seed)

    def _split(seqs, y):
        X = build_profile_features(seqs)
        idx = rng.permutation(len(X))
        n_val = int(len(X) * val_frac)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    Xtr_v, ytr_v, Xva_v, yva_v = _split(seqs_viab, y_viab)
    Xtr_s, ytr_s, Xva_s, yva_s = _split(seqs_sel,  y_sel)

    if verbose:
        print(f"Viability   dataset: {seqs_viab.shape[0]} pairs -> {len(Xtr_v)} train / {len(Xva_v)} val")
        print(f"Selectivity dataset: {seqs_sel.shape[0]} pairs -> {len(Xtr_s)} train / {len(Xva_s)} val")

    state_viab, hist_viab = train_profile_mlp(Xtr_v, ytr_v, Xva_v, yva_v, seed=seed, verbose=verbose, **mlp_kwargs)
    state_sel,  hist_sel  = train_profile_mlp(Xtr_s, ytr_s, Xva_s, yva_s, seed=seed, verbose=verbose, **mlp_kwargs)

    F_viab_hat = extract_effective_F(state_viab, protocol._T_viab)
    F_sel_hat  = extract_effective_F(state_sel,  protocol._T_sel)

    info = dict(
        model_viab=state_viab, model_sel=state_sel,
        history_viab=hist_viab, history_sel=hist_sel,
        n_obs_viab=Xtr_v.shape[0] + Xva_v.shape[0],
        n_obs_sel=Xtr_s.shape[0] + Xva_s.shape[0],
    )
    return F_viab_hat, F_sel_hat, info


def evaluate_profile_recovery(protocol, F_viab_hat, F_sel_hat):
    """
    Same idea as RegressionV1.evaluate_recovery, but the ground truth is the
    profile-ONLY score (F alone, J zeroed out) since single-site features are all
    this V1 model can possibly learn from.
    """
    v_gt  = np.array(jnp.sum(protocol.F_viab[protocol.sequence, jnp.arange(L)], axis=1))
    s_gt  = np.array(jnp.sum(protocol.F_sel[protocol.sequence,  jnp.arange(L)], axis=1))
    v_hat = np.array(jnp.sum(F_viab_hat[protocol.sequence, jnp.arange(L)], axis=1))
    s_hat = np.array(jnp.sum(F_sel_hat[protocol.sequence,  jnp.arange(L)], axis=1))

    combined_gt  = v_gt  / protocol._T_viab + s_gt  / protocol._T_sel
    combined_hat = v_hat / protocol._T_viab + s_hat / protocol._T_sel

    return dict(
        r_viab_scores      = pearson(v_gt, v_hat),
        r_sel_scores       = pearson(s_gt, s_hat),
        r_viab_weights     = pearson(np.array(protocol.F_viab).ravel(), np.array(F_viab_hat).ravel()),
        r_sel_weights      = pearson(np.array(protocol.F_sel).ravel(),  np.array(F_sel_hat).ravel()),
        precision_at_1pct  = precision_at_k(combined_gt, combined_hat, k_frac=0.01),
        precision_at_10pct = precision_at_k(combined_gt, combined_hat, k_frac=0.10),
    )
