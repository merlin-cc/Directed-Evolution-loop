import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

message = "file 2.1"

### -------- Variables -------- ###
rho   = 1.e-3
alpha = 4000

epsilon = 0.01
T_sel   = 0.5    # selectivity temperature
T_viab  = 0.50    # viability temperature (larger = more uniform production)
M       = 40


###################################################################################################################
###################################### These parameters depend on the libaray #####################################
###################################################################################################################
K_MM    = 1e6    # saturation scale for the NGS pipeline's PCR step -- must track the typical
                 # pool size right after pipetting (~N_pcr), not an arbitrary constant
K_MM_amp= 1e8    # saturation scale for bacterial_amplification -- must track the typical
                 # total abundance of lambda3, not the NGS pipeline's K_MM
###################################################################################################################
###################################################################################################################


N1      = 10_000_000
D       = 10_000_000_000
phi     = 0.1    # NB dispersion for sequencing reads (smaller phi = closer to Poisson)

class Lambda():
    def __init__(self, pool, dilution_factor, D, key, K_MM = K_MM) -> None:
        """
        pool  :  (num_sequences,) - abundance/count vector this pipeline
                                    samples/amplifies/sequences; only relative
                                    proportions matter here. NOT the amino-acid
                                    identity matrix (that's Protocol.sequence,
                                    a bare array — compute_score is the only
                                    thing that needs identity, and it never
                                    goes through this class)
        diltuion factor : int     - Dilution factor for the sampling
        D     :  int              - Sequencing depth
        """
        self.pool   = pool
        self.dilution_factor = dilution_factor
        self.D      = float(D)
        self.key    = key
        self.M      = M
        self.K_MM   = K_MM
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
        """
        proportions = self.pool / jnp.sum(self.pool)
        return jax.random.poisson(self._next_key(), proportions * self.N1)
        """
        N_sample = jnp.sum(self.pool)/self.dilution_factor #dilution by 10
        proportions = self.pool / jnp.sum(self.pool)
        return jax.random.poisson(self._next_key(), proportions * N_sample)
        
        
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
        for _ in tqdm(range(self.M), desc="PCR cycles", leave=False):
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
    def __init__(self, N0, N1, dilution_factor, sequences, D, F_viab, J_viab, F_sel, 
                 J_sel, noise_viab, noise_sel, alpha = alpha, 
                 rho = rho, T_viab = T_viab, T_sel = T_sel, M = M, 
                 K_MM = K_MM) -> None:
        """ 
        This class defined method for each block of the road map
        
        Variables :
        - N0                : number of sequences in the initial library
        - N1                : initial number of sequences sampled
        - dilution factor   : factor of dilution for sampling step in NGS
        - sequences         : (num_sequences, 7) list of sequences in the library
        - d0                : initial diversity (set by sequences.shape[0])
        - lambda0           : initial library (fully deterministic)
        - D                 : depth for NGS
        - F_viab or _sel    : profile scores
        - J_viab or _sel    : pott's scores
        - noise_viab / _sel : noise Z in viability and selectivity steps
        - alpha             : number of capsids per HEK cell transfected
        - rho               : number of HEK cells per plasmids transfected
        - T_viab or sel     : viability or selectivity pressure
        - M                 : number of PCR cycle
        - K_MM              : michaelis constant for PCR
        """
        self.key        = jax.random.key(42)
        self.model      = "Potts"
        self.N0         = N0
        self.N1         = N1
        self.dilution_factor = dilution_factor
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
        self._alpha     = alpha
        self._rho       = rho
        self._T_viab    = T_viab
        self._T_sel     = T_sel
        self._M         = M
        self._K_MM      = K_MM
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
        pipeline      = Lambda(_lambda, self.dilution_factor, self.D, self._next_key(), K_MM=self._K_MM)
        pipeline.pool = pipeline.sample_sequences().astype(jnp.float32)
        pipeline.pool = pipeline.pcr_amplification()
        return pipeline.sequence_reads()
    
    def initial_sampling(self) -> jax.Array:
        mu_1 = self.N1 * self.lambda0 / jnp.sum(self.lambda0)
        self.lambda1 = jax.random.poisson(self._next_key(), mu_1)
        return self.lambda1
    
    def produce_capsids(self) -> jax.Array:
        '''
        Old version, kept for reference (identical distribution to the new one below,
        but materializes rate and V as full (d0, max_cells) arrays -- expensive once
        max_cells grows large, e.g. once diversity collapses in later N_loop_DE rounds
        and abundance concentrates onto a few surviving sequences):

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
        '''

        """
        Poisson-additivity rewrite: conditional on the per-cell noises Z_{s,j}, the
        sum of C_s independent Poisson(rate_{s,j}) draws is EXACTLY Poisson(sum of
        rate_{s,j}) -- same distribution as the old version above, but only Z/E stay
        (d0, max_cells); rate and V collapse to a single (d0,) draw, cutting the
        number of large dense arrays alive at once roughly in half.

        Variables :
        - C          : (d0,) number of transfected HEK cells per sequence, C_s ~ Poisson(rho * lambda1(s))
        - Z          : (d0, max_cells) raw per-cell noise, Z_{s,j} ~ N(0,1)
        - E          : (d0, max_cells) per-cell expression multiplier, E_{s,j} = exp(noise_viab * Z_{s,j}), masked to 0 past C_s
        - mask       : (d0, max_cells) keeps only the C_s real cells per sequence, discards padding
        - total_rate : (d0,) alpha * exp(score(s)/T_viab) * sum_j E_{s,j} -- total Poisson rate for sequence s
        - lambda2    : (d0,) single Poisson draw per sequence, Poisson(total_rate(s))
        """
        scores    = self.compute_score(self.F_viab, self.J_viab)
        C         = jax.random.poisson(self._next_key(), self._rho * self.lambda1.astype(jnp.float32))
        max_cells = int(jnp.max(C)) + 1
        cell_idx  = jnp.arange(max_cells)
        mask      = cell_idx[None, :] < C[:, None]

        Z          = jax.random.normal(self._next_key(), shape=(self.d0, max_cells))
        E          = jnp.where(mask, jnp.exp(self.noise_viab * Z), 0.0)
        total_rate = self._alpha * jnp.exp(scores / self._T_viab) * jnp.sum(E, axis=1)

        self.lambda2 = jax.random.poisson(self._next_key(), total_rate).astype(jnp.float32)
        return self.lambda2
    
    def selectivity(self) -> jax.Array:
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
        pipeline      = Lambda(self.lambda3, self.N1, self.D, self._next_key(), K_MM = K_MM_amp)
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
        self.lambda1  = self.initial_sampling()
        self.lambda2  = self.produce_capsids()
        self.lambda2p = self.NGS(self.lambda2)
        self.lambda3  = self.selectivity()
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
        for _ in tqdm(range(number_of_loop), desc="Directed evolution rounds"):
            lambdas.append(self.loop_DE())
            self.lambda0 = self.lambda4
        return lambdas


class ProtocolBacterialCFU(Protocol):
    """
    Protocol subclass that replaces bacterial_amplification() with a mass/CFU-based
    model instead of the Michaelis-Menten PCR-style one, using the same signature so
    loop_DE() picks it up automatically (dynamic dispatch on self.bacterial_amplification()).

    Assumption (per spec): every plasmid molecule ends up in some bacterium -- no
    plasmid DNA is discarded at the transformation step. The gap between the plasmid
    count and the resulting number of transformed bacteria (cfu) is because several
    plasmid copies of the same sequence co-transform a single cell, not because DNA
    is lost. This model is fully deterministic (no Poisson noise): both the mass
    conversion and the 36 division cycles are exact multiplications, so lambda4 ends
    up an EXACT linear rescaling of lambda3 -- relative proportions between variants
    are perfectly preserved (Spearman correlation = 1.0 up to the sub-1 threshold
    below), only the absolute scale changes. If you want the bacterial step to
    reshuffle relative abundances again, this model needs stochastic noise added
    (e.g. Poisson cfu counts, or per-cycle Poisson division), which isn't in the spec
    given.
    """

    CFU_PER_UG       = 5e9    # transformation efficiency: colony-forming units per microgram of plasmid DNA
    PLASMIDS_PER_UG  = 1.15e11  # molecules of plasmid DNA per microgram
    N_DIVISIONS      = 36     # bacterial division cycles after transformation

    def bacterial_amplification(self) -> jax.Array:
        """
        Variables :
        - plasmid_mass_ug : (d0,) mass of plasmid DNA per sequence, lambda3(s) / PLASMIDS_PER_UG
        - cfu             : (d0,) transformed bacteria (founder colonies) per sequence,
                            plasmid_mass_ug(s) * CFU_PER_UG -- thresholded to 0 below 1
                            (a variant can't found fewer than 1 colony), same convention
                            as selectivty()
        - lambda4_raw     : (d0,) pool after N_DIVISIONS deterministic doubling cycles,
                            cfu(s) * 2**N_DIVISIONS -- astronomically large in absolute
                            terms (billions-fold), not usable directly as next round's lambda0
        - lambda4         : (d0,) lambda4_raw diluted back down to N0 total molecules via
                            Poisson sub-sampling (same convention as Lambda.sample_sequences()),
                            so N_loop_DE's successive rounds stay at a stable scale instead
                            of compounding N_DIVISIONS-fold growth every round
        """
        plasmid_mass_ug = self.lambda3 / self.PLASMIDS_PER_UG
        raw_cfu         = plasmid_mass_ug * self.CFU_PER_UG
        cfu             = jnp.where(raw_cfu < 1.0, 0.0, raw_cfu)
        lambda4_raw     = cfu * (2.0 ** self.N_DIVISIONS)

        total       = jnp.sum(lambda4_raw)
        proportions = jnp.where(total > 0, lambda4_raw / total, 0.0)
        self.lambda4 = jax.random.poisson(self._next_key(), proportions * self.N0).astype(jnp.float32)
        return self.lambda4

class ProtocolV2(ProtocolBacterialCFU):
    
    def selectivity(self) -> jax.Array:
        """
        Binomial retention: lambda3(s) ~ Binomial(n=lambda2(s), p=p_s), which
        guarantees lambda3(s) <= lambda2(s) by construction (no per-cell padded
        matrix needed, O(d0) memory instead of O(d0 x max_cells)).

        Variables :
        - scores : (d0,) selectivity score s_sel(s), from compute_score(F_sel, J_sel)
        - p      : (d0,) retention probability, sigmoid(scores(s) / T_sel)
        - lambda3: (d0,) capsid pool after retention, Binomial(lambda2(s), p(s))
        """
        scores       = self.compute_score(self.F_sel, self.J_sel)
        p            = jax.nn.sigmoid(scores / self._T_sel)
        self.lambda3 = jax.random.binomial(self._next_key(), self.lambda2, p).astype(jnp.float32)
        return self.lambda3

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

def initialize_random_weights(key, sparsity_J=0.0):
    """
    Initialize random weights according a given jax key
    F_v = profile weights for viability
    F_s = profile weights for selectivity
    J_v = Pott's weights for viability
    J_s = Pott's weights for viability

    sparsity_J : fraction of the pairwise J entries randomly zeroed out (0 = fully dense,
        the default -- unchanged behavior for existing callers). Real epistatic couplings are
        typically sparse -- most position/amino-acid pairs don't actually interact -- so e.g.
        sparsity_J=0.95 keeps only ~5% of entries nonzero (chosen at random), which is both
        more biologically realistic and gives J a smaller overall magnitude/variance.
    """
    key, key_Fv, key_Fs, key_Jv, key_Js, key_mask_v, key_mask_s = jax.random.split(key, 7)
    num_positions   = 7      # L
    num_amino_acids = 20     # A

    ###############################################
    ### --------------- F score --------------- ###
    ###############################################
    F_v = jax.random.normal(key_Fv, shape=(num_amino_acids, num_positions))
    F_s = jax.random.normal(key_Fs, shape=(num_amino_acids, num_positions))

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
    n_interactions = len(Interactions)
    keep_prob = 1.0 - sparsity_J

    pair_weights_viab = jax.random.normal(key_Jv, shape=(n_interactions,)) / 10
    mask_viab = jax.random.bernoulli(key_mask_v, p=keep_prob, shape=(n_interactions,))
    pair_weights_viab = pair_weights_viab * mask_viab

    J_v = build_J([
        (*ij, float(v))
        for ij, v in zip(Interactions, pair_weights_viab)
    ])

    pair_weights_sel = jax.random.normal(key_Js, shape=(n_interactions,)) / 10
    mask_sel = jax.random.bernoulli(key_mask_s, p=keep_prob, shape=(n_interactions,))
    pair_weights_sel = pair_weights_sel * mask_sel

    J_s = build_J([
        (*ij, float(v))
        for ij, v in zip(Interactions, pair_weights_sel)
    ])

    ###############################################

    return F_v, F_s, J_v, J_s