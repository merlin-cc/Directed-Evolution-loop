import jax
import jax.numpy as jnp
import numpy as np


### -------- Variables -------- ###
rho   = 1.e-3
alpha = 4000

epsilon = 0.01
T_sel   = 0.5    # selectivity temperature
T_viab  = 0.50    # viability temperature (larger = more uniform production)
M       = 20


###################################################################################################################
###################################### These parameters depend on the libaray #####################################
###################################################################################################################
K_MM    = 1e6    # saturation scale for the NGS pipeline's PCR step -- must track the typical
                 # pool size right after pipetting (~N_pcr), not an arbitrary constant
K_MM_amp= 1e8    # saturation scale for bacterial_amplification -- must track the typical
                 # total abundance of lambda3, not the NGS pipeline's K_MM
###################################################################################################################
###################################################################################################################

N_pcr   = 1_000_000
N1      = 10_000_000
D       = 100_000_000
phi     = 0.1    # NB dispersion for sequencing reads (smaller phi = closer to Poisson)

class Lambda():
    def __init__(self, pool, D, key, K_MM = K_MM) -> None:
        """
        pool  :  (num_sequences,) - abundance/count vector this pipeline
                                    samples/amplifies/sequences; only relative
                                    proportions matter here. NOT the amino-acid
                                    identity matrix (that's Protocol.sequence,
                                    a bare array — compute_score is the only
                                    thing that needs identity, and it never
                                    goes through this class)
        D     :  int              - Sequencing depth
        """
        self.pool   = pool
        self.D      = float(D)
        self.key    = key
        self.M      = M
        self.K_MM   = K_MM
        self.N_pcr  = N_pcr
        self.phi    = phi

    def _next_key(self) -> jax.Array:
        """
        Generate a new independant random key
        """
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def sample_sequences(self) -> jax.Array:
        """
        Pipetting is simulated by a Poisson distribution
        """
        proportions = self.pool / jnp.sum(self.pool)
        return jax.random.poisson(self._next_key(), proportions * self.D)

    def sequence_reads(self) -> jax.Array:
        """
        The sequencing is simulated by a Negative Binomial distribution, matching
        sequencing_approx from the notebook (accounts for PCR/sequencing overdispersion
        instead of the pure-Poisson approximation):
            q_s  = pool(s) / sum(pool)
            mu_s = D * q_s
            r    = 1 / phi                 (dispersion, phi fixed)
            p_s  = r / (r + mu_s)
        JAX has no built-in negative-binomial sampler, so this uses the standard
        Gamma-Poisson mixture identity: NB(r, p) = Poisson(Gamma(r, scale=(1-p)/p)).
        """
        proportions = self.pool / jnp.sum(self.pool)
        mu = self.D * proportions
        r  = 1.0 / self.phi
        p  = r / (r + mu)

        key_gamma, key_poisson = self._next_key(), self._next_key()
        gamma_sample = jax.random.gamma(key_gamma, r, shape=mu.shape) * ((1.0 - p) / p)
        return jax.random.poisson(key_poisson, gamma_sample)

    def pcr_amplification(self) -> jax.Array:
        """
        M cycles of PCR amplification with Michaelis-Menten kinetics.
        At each cycle n:
            p_n = 1 / (1 + sum(X) / K_MM)
            X  <- X + Poisson(X * p_n)
        Returns:
        (num_sequences,) float array — amplified pool
        """
        self.pool = self.pool.astype(jnp.float32)
        for _ in range(self.M):
            p_n       = 1.0 / (1.0 + jnp.sum(self.pool) / self.K_MM)
            self.pool = self.pool + jax.random.poisson(self._next_key(), self.pool * p_n).astype(jnp.float32)
        return self.pool

    def ngs(self) -> jax.Array:
        """
        Full NGS process: sampling -> PCR amplification -> sequencing.
        Returns:
        (num_sequences,) int array — read counts
        """
        pool_original = self.pool  # save to restore after ngs mutates self.pool
        self.pool     = self.sample_sequences().astype(jnp.float32)
        self.pool     = self.pcr_amplification()
        result        = self.sequence_reads()
        self.pool     = pool_original  # restore so repeated calls are safe
        return result
    
        

class Protocol():
    def __init__(self, N0, N1, sequences, D, F_viab, J_viab, F_sel, 
                 J_sel, noise_viab, noise_sel, alpha = alpha, 
                 rho = rho, T_viab = T_viab, T_sel = T_sel, M = M, 
                 K_MM = K_MM, N_pcr = N_pcr) -> None:
        """ 
        This class defined method for each block of the road map
        
        Variables :
        - N0            : number of sequences in the initial library
        - N1            : number of sequences sampled
        - sequences     : (num_sequences, 7) list of sequences in the library
        - d0            : Initial diversity
        - lambda0       : initial library (fully deterministic)
        - D             : Depth
        """
        self.key        = jax.random.key(42)
        self.model      = "Potts"
        self.N0         = N0
        self.N1         = N1
        self.sequence   = sequences
        self.d0         = sequences.shape[0]
        self.lambda0    = jnp.full(self.d0, self.N0/self.d0)
        ### ---- other lambda ---- ###
        self.lambda0p   = jnp.zeros(self.d0)
        self.lambda1    = jnp.zeros(self.d0)
        self.lambda2    = jnp.zeros(self.d0)
        self.lambda2p   = jnp.zeros(self.d0)
        self.lambda3    = jnp.zeros(self.d0)
        self.lambda3p   = jnp.zeros(self.d0)
        self.lambda4    = jnp.zeros(self.d0)
        ### ---------------------- ###
        self.D          = float(D)
        self.F_viab     = F_viab
        self.J_viab     = J_viab
        self.F_sel      = F_sel
        self.J_sel      = J_sel
        self.noise_viab = noise_viab
        self.noise_sel  = noise_sel
        ### ---- Other values ---- ###
        self._alpha      = alpha
        self._rho        = rho
        self._T_viab     = T_viab
        self._T_sel      = T_sel
        self._M          = M
        self._K_MM       = K_MM
        self._N_pcr      = N_pcr
        ### ---------------------- ###
    
    def _next_key(self) -> jax.Array:
        """ 
        Generate a new independant random key
        """
        self.key, subkey = jax.random.split(self.key)
        return subkey
    
    def compute_score(self, F, J) -> jax.Array:
        """ 
        Compute scores according the type of model
        """
        if self.model == "Potts":
            scores = jnp.sum(F[self.sequence, jnp.arange(7)], axis =1)
            for i in range(7):
                for j in range(i + 1, 7):
                    scores = scores + J[i, j, self.sequence[:, i], self.sequence[:, j]]
            return jnp.array(scores)
        else:
            return jnp.sum(F[self.sequence, jnp.arange(7)], axis =1)

    def NGS(self, _lambda) -> jax.Array:
        """
        Full NGS process on a pool: pipetting -> PCR amplification -> sequencing,
        delegating to the Lambda class instead of redoing its logic here.

        Variables :
        - pipeline : temporary Lambda wrapping _lambda, reused for each of the 3 steps
        - reads    : (d0,) final sequencing reads
        """
        pipeline      = Lambda(_lambda, self._N_pcr, self._next_key(), K_MM=self._K_MM)
        pipeline.pool = pipeline.sample_sequences().astype(jnp.float32)
        pipeline.pool = pipeline.pcr_amplification()
        pipeline.D    = self.D
        return pipeline.sequence_reads()
    
    def sampling(self) -> jax.Array:
        mu_1 = self.N1 * self.lambda0 / jnp.sum(self.lambda0)
        self.lambda1 = jax.random.poisson(self._next_key(), mu_1)
        return self.lambda1
    
    def produce_capsids(self) -> jax.Array:
        """ 
        HEK transfection and per-cell capsid production, modulated by viability score

        Variables :
        - C : (d0,) number of transfected HEK cells per sequence, C_s ~ Poisson(rho * lambda1(s))
        - Z : (d0, max_cells) raw per-cell noise, Z_{s,j} ~ N(0,1)
        - E : (d0, max_cells) per-cell expression multiplier, E_{s,j} = exp(noise_viab * Z_{s,j})
        - V : (d0, max_cells) capsids from a single cell, V_{s,j} ~ Poisson(alpha * E_{s,j} * exp(score(s)/T_viab))
        - mask : (d0, max_cells) keeps only the C_s real cells per sequence, discards padding
        """
        scores    = self.compute_score(self.F_viab, self.J_viab)
        C         = jax.random.poisson(self._next_key(), self._rho * self.lambda1.astype(jnp.float32))
        max_cells = int(jnp.max(C)) + 1
        cell_idx  = jnp.arange(max_cells)
        mask      = cell_idx[None, :] < C[:, None]

        Z       = jax.random.normal(self._next_key(), shape=(self.d0, max_cells))
        E       = jnp.exp(self.noise_viab * Z)
        rate    = self._alpha * E * jnp.exp(scores / self._T_viab)[:, None]
        V       = jax.random.poisson(self._next_key(), rate)

        self.lambda2 = jnp.sum(jnp.where(mask, V, 0), axis=1)
        return self.lambda2
    
    def selectivty(self) -> jax.Array:
        """ 
        Selectivity (retention) step: enrich the capsid pool by its noisy selectivity score

        Variables :
        - scores       : (d0,) selectivity score s_sel(s), from compute_score(F_sel, J_sel)
        - Z            : (d0,) raw per-sequence noise, Z(s) ~ N(0,1)
        - noisy_scores : (d0,) noisy selectivity score, scores(s) + noise_sel * Z(s)
        - lambda3      : (d0,) capsid pool after retention, lambda2(s) * exp(noisy_scores(s) / T_sel),
                         thresholded to 0 below 1 (a variant can't have fewer than 1 copy)
        """
        scores  = self.compute_score(self.F_sel, self.J_sel)
        Z       = jax.random.normal(self._next_key(), scores.shape)
        noisy_scores = scores + self.noise_sel * Z
        raw_lambda3  = self.lambda2 * jnp.exp(noisy_scores / self._T_sel)
        self.lambda3 = jnp.where(raw_lambda3 < 1.0, 0.0, raw_lambda3)
        return self.lambda3

    def bacterial_amplification(self) -> jax.Array:
        """
        Bacterial amplification (e.g. re-transformation + colony growth): same
        Michaelis-Menten kinetics as PCR amplification, delegated to
        Lambda.pcr_amplification() and applied to the post-selectivity pool lambda3.

        Variables :
        - pipeline : temporary Lambda wrapping lambda3, reused only for pcr_amplification()
        - lambda4  : (d0,) pool after M growth cycles, saturating at K_MM (same formula as PCR)
        """
        pipeline      = Lambda(self.lambda3, self.D, self._next_key(), K_MM = K_MM_amp)
        pipeline.pool = self.lambda3.astype(jnp.float32)
        self.lambda4  = pipeline.pcr_amplification()
        return self.lambda4

    def loop_DE(self) -> list[list[jax.Array]]:
        """ 
        Runs one full directed-evolution round: sampling -> capsid production -> selectivity,
        with NGS taken as a side-channel measurement at 3 checkpoints (never fed back in)

        Variables :
        - lambda0  : (d0,) initial library, unchanged (deterministic)
        - lambda1  : (d0,) sampled plasmid pool, from sampling() — uses true lambda0, not lambda0p
        - lambda2  : (d0,) capsid pool after transfection/production, from produce_capsids()
        - lambda3  : (d0,) capsid pool after selectivity retention, from selectivty()
        - lambda0p : (d0,) NGS reads of lambda0 (first NGS checkpoint)
        - lambda2p : (d0,) NGS reads of lambda2 (second NGS checkpoint)
        - lambda3p : (d0,) NGS reads of lambda3 (third NGS checkpoint)

        Returns : [[lambda0, lambda1, lambda2, lambda3], [lambda0p, lambda2p, lambda3p]]
                (true-pool row, NGS-read row — mirrors the two-row protocol diagram)
        """
        self.lambda0  = self.lambda0
        self.lambda0p = self.NGS(self.lambda0)
        self.lambda1  = self.sampling()
        self.lambda2  = self.produce_capsids()
        self.lambda2p = self.NGS(self.lambda2)
        self.lambda3  = self.selectivty()
        self.lambda3p = self.NGS(self.lambda3)
        self.lambda4  = self.bacterial_amplification()
        return [[self.lambda0, self.lambda1, self.lambda2, self.lambda3, self.lambda4], [self.lambda0p, self.lambda2p, self.lambda3p]]
    
    def N_loop_DE(self, number_of_loop) -> list[list[jax.Array]]:
        """ 
        Runs loop_DE() repeatedly, re-seeding lambda0 from the previous round's
        bacterial-amplified pool (lambda4) before each new round

        Variables :
        - lambdas : list of length number_of_loop; each entry is the [bio_row, ngs_row]
                    pair returned by loop_DE() for that round

        Returns : lambdas
        """
        self.lambda0 = jnp.full(self.d0, self.N0/self.d0)
        lambdas = []
        for _ in range(number_of_loop):
            lambdas.append(self.loop_DE())
            self.lambda0 = self.lambda4
        return lambdas
    
###################################################################################################################
###################################################################################################################
###################################################################################################################
############################################### Initializing weights ##############################################
###################################################################################################################
###################################################################################################################
###################################################################################################################

def build_J(interactions):
    """
    Build a symmetric (L, L, A, A) coupling tensor from a list of
    (i, j, a, b, value) entries.  Both (i,j,a,b) and (j,i,b,a) are set.
    """
    A, L = 20, 7
    J = np.zeros((L, L, A, A))
    for (i, j, a, b, v) in interactions:
        J[i, j, a, b] += v
        J[j, i, b, a] += v
    return jnp.array(J)

def initialize_random_weights(key):
    """ 
    Initialize random weights according a given jax key
    F_v = profile weights for viability
    F_s = profile weights for selectivity
    J_v = Pott's weights for viability
    J_s = Pott's weights for viability
    """
    key, key_Fv, key_Fs, key_Jv, key_Js = jax.random.split(key, 5)
    num_positions   = 7      # L
    num_amino_acids = 20     # A
    
    ###############################################
    ### --------------- F score --------------- ###
    ###############################################
    F_v = 0.3*jax.random.normal(key_Fv, shape=(num_amino_acids, num_positions))
    F_s = 0.3*jax.random.normal(key_Fs, shape=(num_amino_acids, num_positions))
    
    ###############################################

    ###############################################
    ### ------- Pairwise interactions J ------- ###
    ###############################################
    Interactions = [
        [i, j, a, b]
        for i in range(num_positions)
        for j in range(i + 1, num_positions)   # i < j to avoid doubling value
        for a in range(num_amino_acids)
        for b in range(num_amino_acids)
    ]

    pair_weights_viab = jax.random.normal(key_Jv, shape=(len(Interactions),))/10

    J_v = build_J([
        (*ij, float(v))
        for ij, v in zip(Interactions, pair_weights_viab)
    ])

    pair_weights_sel = jax.random.normal(key_Js, shape=(len(Interactions),))/10

    J_s = build_J([
        (*ij, float(v))
        for ij, v in zip(Interactions, pair_weights_sel)
    ])
    
    ###############################################
    
    return F_v, F_s, J_v, J_s