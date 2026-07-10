import jax
import jax.numpy as jnp


# ── Score model ──────────────────────────────────────────────────────────────

def calculate_scores(seqs, w):
    """
    Linear score: s(a1,...,aL) = sum_i w[a_i, i]

    Parameters
    ----------
    seqs : (num_sequences, L) int array
    w    : (num_amino_acids, L) float array

    Returns
    -------
    (num_sequences,) float array
    """
    return jnp.sum(w[seqs, jnp.arange(seqs.shape[1])], axis=1)


# ── Sampling and NGS helpers ─────────────────────────────────────────────────

def sample_sequences(X, D, key):
    """Poisson approximation to multinomial sampling: ~ Poisson(D * X[s] / sum(X))."""
    proportions = X / jnp.sum(X)
    return jax.random.poisson(key, proportions * D)


def ngs_simple(X, D, N_pcr, key):
    """Simple Poisson NGS: sampling -> sequencing, no PCR."""
    seq       = Sequence(X, N_pcr, key)
    X_sampled = seq.sample_sequences().astype(jnp.float32)
    reader    = Sequence(X_sampled, D, seq.key)
    return reader.sequence_reads()


def ngs_full(X, D, N_pcr, key, M, K_MM):
    """Full NGS pipeline: sampling -> PCR amplification -> sequencing."""
    pipeline = Sequence_NGS_pipeline(X, D, key, M, K_MM, N_pcr)
    return pipeline.ngs()


# ── Capsid production ────────────────────────────────────────────────────────

def produce_capsids(x1, viability_scores, alpha, rho, T_viab, key):
    """
    HEK transfection and capsid production modulated by viability score.
    P(s) ~ Poisson(rho * x1[s] * alpha * exp(s_viab[s] / T_viab))

    Parameters
    ----------
    x1               : (num_sequences,) float array — plasmid counts
    viability_scores : (num_sequences,) float array
    alpha            : float — capsid yield per transfected cell (~4000)
    rho              : float — HEK cells transfected per plasmid (~1.2e-4)
    T_viab           : float — viability temperature
    key              : JAX random key

    Returns
    -------
    (num_sequences,) int array — capsid counts P(s)
    """
    # Numerically stable: subtract max before exp to avoid float32 overflow
    log_viab = viability_scores / T_viab
    # log_viab = log_viab - jnp.max(log_viab)
    mu = rho * x1.astype(jnp.float32) * alpha * jnp.exp(log_viab)
    return jax.random.poisson(key, mu)


# ── Full protocol ────────────────────────────────────────────────────────────
key = jax.random.key(42)
rho   = 1.2e-4
alpha = 4000

epsilon = 0.3
T_sel   = 0.5    # selectivity temperature
T_viab  = 1.0    # viability temperature (larger = more uniform production)
M       = 5
K_MM    = 1000.0
N_pcr   = 100_000
D       = 50_000
N1      = D

def protocol(n0, sequences, viability_weights, selectivity_weights,
             epsilon_viability = epsilon, epsilon_selectivity = epsilon, 
             alpha = alpha, rho = rho, T_sel = T_sel, T_viab = T_viab, M = M, 
             K_MM = K_MM, D = D, N_pcr = N_pcr, N1 = N1, key = key,
             use_full_ngs=True):
    """
    Full AAV selection protocol simulation.

    Production phase
    ----------------
    1. NGS on X0                             -> lambda0
    2. Sample X0 -> X1                       -> x1
    3. HEK transfection + capsid production  -> capsids  (viability-dependent)
    4. NGS on capsid pool                    -> lambda1p

    Selectivity phase
    -----------------
    5. Noisy selectivity score: s_tilde = s_sel + epsilon * N(0,1)
    6. Softmax enrichment on lambda1p
    7. Final sequencing                      -> n2

    Parameters
    ----------
    viability_weights   : (num_amino_acids, L)
    selectivity_weights : (num_amino_acids, L)
    T_sel               : float — selectivity temperature
    T_viab              : float — viability temperature
    use_full_ngs        : bool  — True: PCR pipeline, False: simple Poisson

    Returns
    -------
    lambda0, capsids, lambda1p, n2
    """
    key, k_ngs0, k_sample, k_capsids, k_ngs1, k_noise, k_ngs2 = jax.random.split(key, 7)

    viability_scores   = calculate_scores(sequences, viability_weights)
    selectivity_scores = calculate_scores(sequences, selectivity_weights)

    if use_full_ngs:
        _ngs = lambda X, k: ngs_full(X, D, N_pcr, k, M, K_MM)
    else:
        _ngs = lambda X, k: ngs_simple(X, D, N_pcr, k)

    # Production
    lambda0  = _ngs(n0, k_ngs0).astype(jnp.float32)
    x1       = sample_sequences(lambda0, N1, k_sample).astype(jnp.float32)
    capsids  = produce_capsids(x1, viability_scores, alpha, rho, T_viab, k_capsids).astype(jnp.float32)
    lambda1p = _ngs(capsids, k_ngs1).astype(jnp.float32)

    # Selectivity — numerically stable softmax (subtract max before exp)
    s_tilde     = selectivity_scores + epsilon * jax.random.normal(k_noise, shape=selectivity_scores.shape)
    log_weights = s_tilde / T_sel - jnp.max(s_tilde / T_sel)
    post_select = lambda1p * jnp.exp(log_weights)
    n2_prime    = post_select / jnp.sum(post_select) * D
    n2          = jax.random.poisson(k_ngs2, n2_prime).astype(jnp.float32)

    return lambda0, capsids, lambda1p, n2


class Sequence():
    def __init__(self, X, D, key) -> None:
        """
        X   : (num_sequences,) float array — pool counts
        D   : int — sequencing depth
        key : JAX random key
        """
        self.X   = X
        self.D   = D
        self.key = key

    def _next_key(self) -> jax.Array:
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def sample_sequences(self) -> jax.Array:
        proportions = self.X / jnp.sum(self.X)
        return jax.random.poisson(self._next_key(), proportions * self.D)

    def sequence_reads(self) -> jax.Array:
        proportions = self.X / jnp.sum(self.X)
        return jax.random.poisson(self._next_key(), proportions * self.D)


class Sequence_NGS_pipeline(Sequence):
    def __init__(self, X, D, key, M, K_MM, N_pcr) -> None:
        super().__init__(X, D, key)
        self.M     = M
        self.K_MM  = K_MM
        self.N_pcr = N_pcr

    def pcr_amplification(self) -> jax.Array:
        """
        M cycles of PCR amplification with Michaelis-Menten kinetics.

        At each cycle n:
            p_n = 1 / (1 + sum(X) / K_MM)
            X  <- X + Poisson(X * p_n)

        Returns
        -------
        (num_sequences,) float array — amplified pool
        """
        self.X = self.X.astype(jnp.float32)
        for _ in range(self.M):
            p_n    = 1.0 / (1.0 + jnp.sum(self.X) / self.K_MM)
            self.X = self.X + jax.random.poisson(self._next_key(), self.X * p_n).astype(jnp.float32)
        return self.X

    def ngs(self) -> jax.Array:
        """
        Full NGS process: sampling -> PCR amplification -> sequencing.

        Returns
        -------
        (num_sequences,) int array — read counts
        """
        X_original = self.X  # save to restore after ngs mutates self.X
        self.X = self.sample_sequences().astype(jnp.float32)
        self.X = self.pcr_amplification()
        result = self.sequence_reads()
        self.X = X_original  # restore so repeated calls are safe
        return result


