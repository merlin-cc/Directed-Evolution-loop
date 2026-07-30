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
#####################################################################################
#####################################################################################
#####################################################################################

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from sequence_classesV1 import *
from analysisV1 import pearson, precision_at_k
from RegressionV1 import build_multi_round_dataset

message = "file MLP regression 1.0"

L = 7   # num_positions, matches Protocol.compute_score's hardcoded range(7)
A = 20  # num_amino_acids

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    # fp32 matmuls (e.g. BatchNorm running-stat updates) run at TF32 instead of full
    # precision -- irrelevant to accuracy here, free throughput on the GB10's tensor cores
    torch.set_float32_matmul_precision("high")


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


def _to_device_f32(x, device):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


### ---------------------------- Model --------------------------- ###
#######################################################################################

class ProfileMLP(nn.Module):
    """
    MLP over single-site one-hot features -> scalar score. Mirrors what Ridge's
    F_flat @ x does for the profile model, but replaces the closed-form L2 solve
    (which collapses toward near-zero weights once the feature count balloons, see
    file header) with a network trained by SGD.
    """

    def __init__(self, input_dim=L * A, hidden_dims=(256, 128, 64), dropout=0.1):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


### ---------------------------- Training --------------------------- ###
#######################################################################################

def train_profile_mlp(X_train, y_train, X_val, y_val, hidden_dims=(256, 128, 64),
                       dropout=0.1, epochs=300, batch_size=4096, lr=1e-3,
                       weight_decay=1e-4, patience=20, device=DEVICE, seed=0, verbose=True):
    """
    GPU-resident training: the whole profile dataset (N x L*A floats -- a few
    hundred MB even at DE_loopV1.ipynb's largest library, 800k sequences) fits
    comfortably in a DGX Spark's unified memory, so X/y are moved to `device` ONCE
    and every epoch's minibatches are carved out with a GPU-side torch.randperm --
    no DataLoader / per-step host<->device copy, which is where a conventional
    discrete-GPU training loop spends most of its time on a model this small.
    Mixed precision uses bf16 (Blackwell's native compute dtype, no GradScaler
    needed the way fp16 requires).

    Returns
    -------
    model   : the ProfileMLP with the best-val-MSE weights loaded
    history : dict with "train_loss" / "val_loss" per epoch
    """
    torch.manual_seed(seed)
    model = ProfileMLP(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(patience // 3, 1))
    loss_fn = nn.MSELoss()

    Xtr, ytr = _to_device_f32(X_train, device), _to_device_f32(y_train, device)
    Xva, yva = _to_device_f32(X_val, device),   _to_device_f32(y_val, device)
    n_train  = Xtr.shape[0]
    use_amp  = device.type == "cuda"

    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in tqdm(range(epochs), desc="Training ProfileMLP", disable=not verbose):
        model.train()
        perm = torch.randperm(n_train, device=device)
        running, seen = 0.0, 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            if idx.numel() < 2:  # BatchNorm1d needs >1 sample in train mode
                continue
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * idx.numel()
            seen    += idx.numel()
        train_loss = running / seen

        model.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            val_loss = loss_fn(model(Xva), yva).item()
        sched.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val MSE={best_val:.4f})")
                break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_scores(model, X, device=DEVICE):
    """Forward pass over the full X in one shot (profile features are only L*A wide)."""
    model.eval()
    Xt = _to_device_f32(X, device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        pred = model(Xt)
    return pred.float().cpu().numpy()


@torch.no_grad()
def extract_effective_F(model, T, L=L, A=A, device=DEVICE):
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

    scores     = predict_scores(model, np.concatenate([base, mutants], axis=0), device=device)
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
        info : dict with the trained models, loss histories, and dataset sizes
    """
    mlp_kwargs = mlp_kwargs or {}

    if verbose:
        gpu_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
        print(f"torch device: {DEVICE} ({gpu_name})")

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

    model_viab, hist_viab = train_profile_mlp(Xtr_v, ytr_v, Xva_v, yva_v, seed=seed, verbose=verbose, **mlp_kwargs)
    model_sel,  hist_sel  = train_profile_mlp(Xtr_s, ytr_s, Xva_s, yva_s, seed=seed, verbose=verbose, **mlp_kwargs)

    F_viab_hat = extract_effective_F(model_viab, protocol._T_viab)
    F_sel_hat  = extract_effective_F(model_sel,  protocol._T_sel)

    info = dict(
        model_viab=model_viab, model_sel=model_sel,
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
