"""
DRAFT -- not integrated, not tested, not imported anywhere. Sketch to review before
picking a direction and folding it into sequence_classesV1.py properly.

Both classes below extend ProtocolV3 (the only pipeline step touched is produce_capsids(),
which ProtocolV3 inherits unchanged from Protocol) to model cross-packaging: non-viable
variants getting a nonzero NGS signal because a co-transfected, functional neighbor's
capsid proteins accidentally encapsidate the non-viable variant's plasmid DNA in the same
HEK cell. This is the mechanism behind the bimodal log2(Production1/2) distribution seen in
fit4functionaav9.csv (reproductibility/log_enrichment_histograms.ipynb) -- the left mode
never fully collapses to -inf the way a pure "dead variants produce zero capsids" model
would predict.

Key structural problem: Protocol.produce_capsids() draws C_s ~ Poisson(rho * lambda1(s))
INDEPENDENTLY per sequence (sequence_classesV1.py:296) -- the C_s cells "belonging" to
sequence s are disjoint from every other sequence's cells by construction. No two different
sequences ever share a simulated cell today, so there is currently no substrate for
cross-packaging to act on at all. A fully mechanistic fix (shared physical cell pool, all
sequences' plasmids competing for the same cells) isn't attempted here -- at this project's
usual N1 scale (rho * N1 reaches ~1e8-1e10 in several sweeps already in this repo,
mu_HEK_multiplicity_sweep.ipynb/diversity_sweep.ipynb family) explicit per-cell occupancy
isn't tractable to materialize.

    ProtocolCrossPackagingBackground -- cheaper alternative: adds a population-level
        "leakage" term to produce_capsids()'s Poisson rate instead of simulating cell
        occupancy. Every sequence's rate gets += cross_packaging_rate * (median rate among
        transfected, i.e. "healthy neighbor") -- a non-viable variant (own rate ~ 0) still
        ends up with a nonzero floor. Stays O(d0), same asymptotic cost as the existing
        produce_capsids().

A THIRD, separate noise source lives below (LambdaWithHallucination /
ProtocolWithHallucination): "new variant appearance" at the NGS measurement step itself,
not at production. See notebooks/reproductibility/new_variant_appearance_analysis.ipynb --
a small fraction of fit4functionaav9.csv variants are exactly zero in one replicate and
clearly in the "fit" population in the other, far beyond normal replicate noise (100% of
flagged rows exceed the p99 of ordinary disagreement). Working hypothesis there: a PCR/
oligo-synthesis error changes one codon construct's actual DNA away from its intended
design, so its reads stop reflecting that sequence's true abundance -- this is a
measurement-layer artifact, independent of cross-packaging (which acts on production,
i.e. lambda2, not on the NGS read-out).

A FOURTH source, also below (LambdaWithPCRMutations / ProtocolWithPCRMutations): point
mutations introduced *during* the amplification PCR rather than pre-existing in the oligo
pool. Each cycle a fraction of the fresh copies take a single-amino-acid change and then
amplify for every REMAINING cycle -- an early error becomes a large jackpot clone, a late
one a single molecule (timing structure LambdaWithHallucination, a pure readout term,
cannot produce).

Because a mutation is a *single* AA change, every 7-mer it can produce is enumerable up
front: the L*(A-1) single substitutions of each designed variant. So the simulated pool is
EXTENDED to `d0_ext` slots -- `[:d0]` the designed library, `[d0:]` every distinct
single-mutant 7-mer that is NOT in the design -- and each PCR error is deposited onto the
slot of the exact 7-mer it spells, where it then amplifies. Two runs share this fixed slot
space, so NGS reads can be compared sequence by sequence (which is the point: novel mutants
you know are errors because you never put them in; but a mutant that happens to spell
another designed variant is indistinguishable and silently perturbs that variant's count).
`ProtocolWithPCRMutations.ext_sequences` (d0_ext, L) is the 7-mer of every slot.

The DE-loop's biological pool is untouched: mutants live only in the throwaway measurement
pipeline (depletion is `_lambda - sampled`, real molecules only). The NGS reads
(lambda0p/lambda2p/lambda3p) become length d0_ext -- slice `[:d0]` to compare against the
designed-only real data. Like hallucination, a measurement-layer artifact independent of
cross-packaging. The two NGS-layer sources compose:
LambdaWithHallucinationAndPCRMutations, and ProtocolCrossPackagingHallucinationAndPCRMutations
stacks all of the above.
"""

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from sequence_classesV1 import Lambda, ProtocolV3

MAX_MECHANISTIC_CELLS = 5_000_000  # well under where a dense (n_cells, d0) occupancy would fit in GPU memory


def _pcr_single_mutant_map(sequence):
    """
    Precompute the fixed slot space for single-amino-acid PCR mutants of the designed
    library.

    Returns:
    - neighbor_idx  : jnp int32 (d0, L, A) -- extended-pool slot of the 7-mer obtained from
                      designed variant i by setting position p to amino acid a. If that
                      7-mer is itself a designed variant -> its designed slot (< d0);
                      otherwise a fresh slot in [d0, d0_ext). For a == the original AA the
                      entry is i (self; it receives zero flux, see is_real_sub).
    - is_real_sub   : jnp bool (d0, L, A) -- True where a != the original AA (the L*(A-1)
                      genuine substitutions; the L self-entries per variant are False).
    - ext_sequences : np.int8 (d0_ext, L) -- the 7-mer of every slot. [:d0] is the designed
                      library as given; [d0:] the novel single-mutants, one row each.
    - d0_ext        : int -- d0 + number of distinct novel single-mutant 7-mers.

    O(d0 * L * A) work and memory (~10 M for a 74 k library over 20**7); a couple of seconds,
    once, at Protocol construction.
    """
    seq   = np.asarray(sequence).astype(np.int64)
    d0, L = seq.shape
    A     = int(seq.max()) + 1
    pw    = A ** np.arange(L)
    codes = seq @ pw                                                    # (d0,)

    nb = np.empty((d0, L, A), np.int64)
    for p in range(L):
        nb[:, p, :] = (codes - seq[:, p] * pw[p])[:, None] + np.arange(A)[None, :] * pw[p]
    is_real = np.arange(A)[None, None, :] != seq[:, :, None]            # (d0, L, A)

    order = np.argsort(codes); csort = codes[order]
    flat  = nb.reshape(-1)
    k     = np.clip(np.searchsorted(csort, flat), 0, d0 - 1)
    is_des = csort[k] == flat
    des_idx = np.where(is_des, order[k], -1)

    novel_codes = np.unique(flat[(~is_des) & is_real.reshape(-1)])
    npos = np.clip(np.searchsorted(novel_codes, flat), 0, max(len(novel_codes) - 1, 0))
    ext_idx = np.where(des_idx >= 0, des_idx, d0 + npos).reshape(d0, L, A)
    ii = np.broadcast_to(np.arange(d0)[:, None, None], (d0, L, A))
    ext_idx = np.where(is_real, ext_idx, ii).astype(np.int32)

    d0_ext = d0 + len(novel_codes)
    ext_sequences = np.zeros((d0_ext, L), np.int8)
    ext_sequences[:d0] = seq.astype(np.int8)
    for p in range(L):
        ext_sequences[d0:, p] = (novel_codes // pw[p]) % A

    return jnp.asarray(ext_idx), jnp.asarray(is_real), ext_sequences, int(d0_ext)


def _run_ngs_pipeline(protocol, _lambda, lambda_cls, **lambda_kwargs):
    """Shared body for the _ngs_and_deplete() overrides that only need to swap the Lambda
    class: a line-for-line copy of Protocol._ngs_and_deplete (sequence_classesV1.py) with
    the Lambda subclass and its extra kwargs parametrised, so the NGS-layer noise sources
    can be mixed. (ProtocolWithHallucination predates this and keeps its own inline copy.)

    The measurement branch and the DE-loop branch stay separate by construction:
    - `pipeline` is a throwaway wrapping the pipetted aliquot; PCR + its mutations live
      entirely inside `pipeline.pool` and never leave this function.
    - `reads` (the NGS observation) is taken from that mutated/amplified aliquot, so it
      carries the PCR-mutation effects (collision jackpots on designed variants + the
      parasitic depth spent on novel mutants).
    - `depleted` = `_lambda - sampled`, i.e. the true pool minus only the *real* molecules
      pipetted out -- PCR mutants are never added back, so the next loop step never sees them.

    The last pipeline object is stashed on `protocol._last_ngs_pipeline` for inspection
    (e.g. `.pcr_novel_mutant_mass`)."""
    pipeline      = lambda_cls(
        _lambda, protocol.dilution_factor, protocol.D, protocol._next_key(),
        multinomialNGS=protocol.multinomialNGS, K_MM=protocol._K_MM, M=protocol._M,
        phi=protocol._phi, **lambda_kwargs,
    )
    sampled       = pipeline.sample_sequences().astype(jnp.float32)
    pipeline.pool = sampled
    pipeline.pool = pipeline.pcr_amplification()
    reads         = pipeline.sequence_reads()
    depleted      = jnp.maximum(_lambda - sampled, 0.0)
    protocol._last_ngs_pipeline = pipeline
    return reads, depleted


class ProtocolCrossPackagingBackground(ProtocolV3):
    """
    produce_capsids() override: population-level leakage floor, no per-cell simulation.

    New parameter:
    - cross_packaging_rate : fraction (>= 0) of the population's mean "healthy" capsid
      rate that leaks into every sequence's own rate. 0 recovers Protocol's exact
      behavior. Free-standing calibration constant for now -- doesn't yet depend on mu
      (higher multiplicity should mean more co-transfection opportunity, so probably
      should scale with mu eventually, cf. discussion).
    """

    def __init__(self, *args, cross_packaging_rate: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cross_packaging_rate = cross_packaging_rate

    def produce_capsids(self) -> jax.Array:
        """
        Same Poisson-additivity formulation as Protocol.produce_capsids(), plus a shared
        leakage term added to every sequence's rate before the final draw.

        Variables :
        - own_rate              : (d0,) exactly Protocol's total_rate -- this sequence's
                                   own transfected-cell capsid production, no leakage
        - transfected           : (d0,) bool, whether this sequence got >= 1 cell (C_s > 0)
        - healthy_reference_rate: scalar, MEDIAN own_rate among transfected sequences --
                                   proxy for "what a co-transfected functional neighbor is
                                   typically producing in a shared cell". Median, not mean:
                                   own_rate is alpha * exp(score / T_viab), so a handful of
                                   high-scoring outliers can be orders of magnitude above
                                   the bulk -- a mean-based floor gets dragged up to an
                                   implausibly high leakage level by those outliers alone
                                   (checked empirically on a random smoke test: mean-based
                                   floor was ~5x the median-based one)
        - leakage               : scalar, cross_packaging_rate * healthy_reference_rate,
                                   added to EVERY sequence regardless of transfection/score
        - total_rate            : (d0,) own_rate + leakage
        """
        scores    = self.compute_score(self.F_viab, self.J_viab)
        C         = jax.random.poisson(self._next_key(), self._rho * self.lambda1.astype(jnp.float32))
        max_cells = int(jnp.max(C)) + 1
        cell_idx  = jnp.arange(max_cells)
        mask      = cell_idx[None, :] < C[:, None]

        Z        = jax.random.normal(self._next_key(), shape=(self.d0, max_cells))
        E        = jnp.where(mask, jnp.exp(self.noise_viab * Z / self._T_viab), 0.0)
        own_rate = self._alpha * jnp.exp(scores / self._T_viab) * jnp.sum(E, axis=1)

        transfected            = C > 0
        healthy_reference_rate = jnp.nanmedian(jnp.where(transfected, own_rate, jnp.nan))
        leakage                = self.cross_packaging_rate * healthy_reference_rate
        total_rate             = own_rate + leakage

        self.lambda2 = jax.random.poisson(self._next_key(), total_rate).astype(jnp.float32)
        return self.lambda2


class LambdaWithHallucination(Lambda):
    """
    Lambda subclass: sequence_reads() gets independent "hallucinated" reads added on top
    of the real NGS signal, for a random subset of sequences, REGARDLESS of their actual
    pool abundance -- see new_variant_appearance_analysis.ipynb. Working hypothesis: a
    PCR/oligo-synthesis error silently changes one codon construct's actual DNA away from
    its intended designed sequence, so the reads counted against that sequence at this
    checkpoint stop reflecting its true abundance (self.pool) at all -- unlike ordinary
    NGS noise (Poisson/NB/multinomial sampling around the true proportions), this is a
    read that shouldn't be attributed to this sequence in the first place.

    New attributes (plain attributes, not constructor-only -- can be flipped after
    construction the same way multinomialNGS already is, e.g. `lambda_obj.hallucination
    = True`):
    - hallucination      : bool, on/off switch. Default False -- no behavior change
                            unless explicitly enabled.
    - hallucination_rate : float, probability that ANY given sequence gets a hallucinated
                            read event at THIS checkpoint (independent draw per checkpoint,
                            so the same sequence can hallucinate at one NGS checkpoint and
                            not another). Default 69/74464 ~= 0.000927 -- the empirical
                            fraction of fit4functionaav9.csv variants flagged by
                            new_variant_appearance_analysis.ipynb as exactly zero in one
                            replicate and clearly "fit" in the other (100% of those rows
                            were beyond the p99 of ordinary replicate disagreement, so this
                            is a real rate estimate, not just a round number).
    """

    def __init__(self, *args, hallucination: bool = False,
                 hallucination_rate: float = 69 / 74_464, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hallucination      = hallucination
        self.hallucination_rate = hallucination_rate

    def sequence_reads(self) -> jax.Array:
        """
        Real reads exactly as Lambda.sequence_reads(), then -- only if self.hallucination
        is True -- each sequence independently has probability hallucination_rate of
        getting a spurious read count added on top.

        Variables :
        - reads              : (d0,) real NGS reads, unchanged from Lambda's own formula
        - hallucinated_mask  : (d0,) bool, which sequences hallucinate at this checkpoint
        - hallucinated_reads : (d0,) Poisson(D / d0) where masked -- "one average
                                detectable variant's share" of the sequencing depth, i.e.
                                enough to land in the fit population rather than a couple
                                of stray reads. Not separately calibrated against real data
                                (only hallucination_rate is) -- a fixed magnitude choice,
                                flagged here for whoever tunes this further.
        """
        reads = super().sequence_reads()
        if not self.hallucination:
            return reads

        d0                  = self.pool.shape[0]
        hallucinated_mask   = jax.random.bernoulli(self._next_key(), self.hallucination_rate, shape=(d0,))
        hallucinated_reads  = jax.random.poisson(self._next_key(), self.D / d0, shape=(d0,))
        return reads + jnp.where(hallucinated_mask, hallucinated_reads, 0)


class ProtocolWithHallucination(ProtocolV3):
    """
    Protocol subclass wiring LambdaWithHallucination into every NGS checkpoint
    (lambda0p/lambda2p/lambda3p via loop_DE()'s _ngs_and_deplete(), and standalone NGS()
    calls, which delegate to the same method) -- _ngs_and_deplete() is otherwise an exact
    copy of Protocol's own, only the Lambda class instantiated changes. Depletion is
    computed from `sampled` (drawn from the real `_lambda`), NOT from the hallucinated
    reads -- a hallucinated read is a measurement artifact, it doesn't correspond to a
    real molecule leaving the true pool.

    New constructor parameters: hallucination (bool, default False), hallucination_rate
    (float, default 69/74464 -- see LambdaWithHallucination). Composable with
    ProtocolCrossPackagingBackground (different pipeline steps: produce_capsids() vs.
    _ngs_and_deplete(), no conflict) -- see ProtocolCrossPackagingAndHallucination below.
    Composes with LambdaWithPCRMutations at the Lambda level (different methods:
    sequence_reads vs pcr_amplification) -- see ProtocolCrossPackagingHallucinationAndPCRMutations.
    """

    def __init__(self, *args, hallucination: bool = False,
                 hallucination_rate: float = 69 / 74_464, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hallucination      = hallucination
        self.hallucination_rate = hallucination_rate

    def _ngs_and_deplete(self, _lambda) -> tuple[jax.Array, jax.Array]:
        pipeline      = LambdaWithHallucination(
            _lambda, self.dilution_factor, self.D, self._next_key(),
            multinomialNGS=self.multinomialNGS, K_MM=self._K_MM, M=self._M, phi=self._phi,
            hallucination=self.hallucination, hallucination_rate=self.hallucination_rate,
        )
        sampled       = pipeline.sample_sequences().astype(jnp.float32)
        pipeline.pool = sampled
        pipeline.pool = pipeline.pcr_amplification()
        reads         = pipeline.sequence_reads()
        depleted      = jnp.maximum(_lambda - sampled, 0.0)
        return reads, depleted


class ProtocolCrossPackagingAndHallucination(ProtocolCrossPackagingBackground, ProtocolWithHallucination):
    """
    Both noise sources together: cross-packaging leakage in produce_capsids() (from
    ProtocolCrossPackagingBackground) AND NGS hallucination in _ngs_and_deplete() (from
    ProtocolWithHallucination). No method conflict -- the two parents override DIFFERENT
    pipeline steps -- so this class needs no body of its own: cooperative __init__ (MRO:
    this class -> ProtocolCrossPackagingBackground -> ProtocolWithHallucination ->
    ProtocolV3 -> ...) sets cross_packaging_rate/hallucination/hallucination_rate all
    correctly, as long as the CALLER passes all three by keyword (they can't be
    positional -- each parent's __init__ only declares its own kwarg explicitly and
    forwards everything else via **kwargs).
    """
    pass


class LambdaWithPCRMutations(Lambda):
    """
    Lambda subclass: point mutations introduced *during* pcr_amplification(). Each cycle a
    fraction `pcr_mutation_rate` of the freshly-synthesised copies of a designed variant take
    ONE random single-amino-acid change; that copy is moved onto the extended-pool slot of
    the 7-mer it now spells (a designed slot if the mutant IS another library member, else a
    dedicated novel slot) and amplifies there for every remaining cycle -- an error in cycle
    1 founds a jackpot clone, one in the last cycle is a lone molecule.

    self.pool is EXTENDED to `pcr_d0_ext` slots on entry: [:d0] the designed library (from
    the pipetted aliquot), [d0:] every distinct novel single-mutant 7-mer, all starting at
    zero. It is returned at that length, so sequence_reads() -> reads of length d0_ext, and
    two runs are directly comparable slot by slot (ext_sequences names every slot). Only the
    d0 designed slots spawn errors (a mutant re-mutating is rate**2, negligible, and it has
    no precomputed neighbour map).

    Constructor kwargs (filled in by ProtocolWithPCRMutations):
    - pcr_mutations     : bool, on/off. Default False -- no behaviour change.
    - pcr_mutation_rate : float, P(a fresh copy takes a single-AA error) per cycle. Default
                          1e-4 -- order-of-magnitude proofreading-free polymerase over the
                          ~21 nt variable region, non-synonymous fraction only. Tune.
    - pcr_neighbor_idx  : int32 (d0, L, A) slot map,   from _pcr_single_mutant_map
    - pcr_is_real_sub   : bool  (d0, L, A) real-sub mask
    - pcr_n_subs        : int, L*(A-1) (133) -- per-substitution split denominator
    - pcr_d0_ext        : int, extended pool length
    """

    def __init__(self, *args, pcr_mutations: bool = False, pcr_mutation_rate: float = 1e-4,
                 pcr_neighbor_idx=None, pcr_is_real_sub=None, pcr_n_subs: int = 133,
                 pcr_d0_ext=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pcr_mutations     = pcr_mutations
        self.pcr_mutation_rate = pcr_mutation_rate
        self.pcr_neighbor_idx  = pcr_neighbor_idx
        self.pcr_is_real_sub   = pcr_is_real_sub
        self.pcr_n_subs        = pcr_n_subs
        self.pcr_d0_ext        = pcr_d0_ext

    def pcr_amplification(self) -> jax.Array:
        """
        Lambda.pcr_amplification() unchanged when pcr_mutations is False. When True: extend
        the pool to pcr_d0_ext, then each of the M cycles

        Variables :
        - births   : (d0_ext,) fresh copies this cycle, Poisson(pool * p_n) -- designed AND
                      novel slots amplify
        - errored  : (d0,) fresh copies of designed variants that took a single-AA change,
                      Poisson(births[:d0] * pcr_mutation_rate) clamped to births -- removed
                      from their source
        - flux     : (d0, L, A) of each variant's errors, how many took substitution
                      (position p -> aa a), Poisson(errored / pcr_n_subs) on the real-sub
                      entries -- scattered onto pcr_neighbor_idx (the spelled 7-mer's slot)
        """
        if not self.pcr_mutations:
            return super().pcr_amplification()

        d0        = self.pool.shape[0]
        ext       = jnp.concatenate(
            [self.pool.astype(jnp.float32), jnp.zeros(self.pcr_d0_ext - d0, jnp.float32)])
        idx_flat  = self.pcr_neighbor_idx.reshape(-1)
        sub_rate  = self.pcr_is_real_sub.astype(jnp.float32) * (1.0 / self.pcr_n_subs)  # (d0,L,A)
        di        = jnp.arange(d0)

        for _ in tqdm(range(self.M), desc="PCR cycles (+mut)", leave=False):
            p_n = 1.0 / (1.0 + jnp.sum(ext) / self.K_MM)

            births  = jax.random.poisson(self._next_key(), ext * p_n).astype(jnp.float32)
            errored = jnp.minimum(
                jax.random.poisson(self._next_key(), births[:d0] * self.pcr_mutation_rate).astype(jnp.float32),
                births[:d0])
            ext = ext + births.at[di].add(-errored)          # designed: +births-errored, novel: +births

            flux = jax.random.poisson(
                self._next_key(), errored[:, None, None] * sub_rate).astype(jnp.float32)   # (d0,L,A)
            ext = ext.at[idx_flat].add(flux.reshape(-1))

        return ext

    def sequence_reads(self) -> jax.Array:
        """
        Lambda.sequence_reads() unchanged when pcr_mutations is False. When True, reads are
        Poisson(D * proportions) over the FULL extended pool (independent per slot): a true
        Multinomial(D, .) over ~1e7 mostly-empty slots is impractical, and the Poisson
        marginal is identical bar reads summing to ~D rather than exactly D. Designed slots
        are naturally deflated by whatever depth the novel slots took.
        """
        if not self.pcr_mutations:
            return super().sequence_reads()
        s = jnp.sum(self.pool)
        if s == 0:
            raise ValueError("No sequence in the pool, thus no NGS can be performed")
        proportions = self.pool / s
        if self.multinomialNGS:
            return jax.random.poisson(self._next_key(), self.D * proportions).astype(jnp.float32)
        mu = self.D * proportions
        r  = 1.0 / self.phi
        p  = r / (r + mu)
        key_gamma, key_poisson = self._next_key(), self._next_key()
        return jax.random.poisson(key_poisson,
                                  jax.random.gamma(key_gamma, r, shape=mu.shape) * ((1.0 - p) / p))


class ProtocolWithPCRMutations(ProtocolV3):
    """
    ProtocolV3 wiring LambdaWithPCRMutations into every NGS checkpoint -- same injection
    point as ProtocolWithHallucination (_ngs_and_deplete via loop_DE()). __init__ runs
    _pcr_single_mutant_map(self.sequence) once (~seconds) and exposes:
    - self.d0_ext        : extended NGS-read length (d0 + #novel single-mutants)
    - self.ext_sequences : np.int8 (d0_ext, L), the 7-mer of every read slot ([:d0] designed)

    loop_DE()'s lambda0p/lambda2p/lambda3p come back length d0_ext -- slice [:d0] to compare
    against designed-only real data; [d0:] is the named novel-mutant spectrum. The DE-loop's
    biological lambdas stay length d0 and mutant-free.

    New params: pcr_mutations (bool, default False), pcr_mutation_rate (float, default 1e-4).
    """

    def __init__(self, *args, pcr_mutations: bool = False, pcr_mutation_rate: float = 1e-4,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pcr_mutations     = pcr_mutations
        self.pcr_mutation_rate = pcr_mutation_rate
        (self._pcr_neighbor_idx, self._pcr_is_real_sub,
         self.ext_sequences, self.d0_ext) = _pcr_single_mutant_map(self.sequence)
        self._pcr_n_subs = self.sequence.shape[1] * (int(np.asarray(self.sequence).max()))

    def _pcr_lambda_kwargs(self) -> dict:
        return dict(pcr_mutations=self.pcr_mutations, pcr_mutation_rate=self.pcr_mutation_rate,
                    pcr_neighbor_idx=self._pcr_neighbor_idx, pcr_is_real_sub=self._pcr_is_real_sub,
                    pcr_n_subs=self._pcr_n_subs, pcr_d0_ext=self.d0_ext)

    def _ngs_and_deplete(self, _lambda) -> tuple[jax.Array, jax.Array]:
        return _run_ngs_pipeline(self, _lambda, LambdaWithPCRMutations, **self._pcr_lambda_kwargs())


class LambdaWithHallucinationAndPCRMutations(LambdaWithHallucination, LambdaWithPCRMutations):
    """
    Both NGS-layer error sources at once: readout hallucination (LambdaWithHallucination
    overrides sequence_reads -- and calls super(), so it stacks on top of PCRMutations'
    extended-pool Poisson reads) + in-PCR mutation jackpots (LambdaWithPCRMutations overrides
    pcr_amplification). Cooperative multiple inheritance, no body needed. Caller passes
    every hallucination_* and pcr_* kwarg by keyword. Note: with PCR mutations on, the
    pool (hence the hallucination mask and the D/d0 magnitude) is length d0_ext, so
    hallucination also acts on the novel-mutant slots.
    """
    pass


class ProtocolCrossPackagingHallucinationAndPCRMutations(
        ProtocolCrossPackagingBackground, ProtocolWithHallucination, ProtocolWithPCRMutations):
    """
    All three noise sources: cross-packaging leakage in produce_capsids() (from
    ProtocolCrossPackagingBackground) + hallucination AND in-PCR mutations at the NGS
    layer. produce_capsids() resolves to ProtocolCrossPackagingBackground's via the MRO;
    _ngs_and_deplete() is overridden here to run the combined
    LambdaWithHallucinationAndPCRMutations (the MRO would otherwise pick
    ProtocolWithHallucination's, which knows nothing about PCR mutations). Cooperative
    __init__ (MRO: this -> CrossPackagingBackground -> WithHallucination -> WithPCRMutations
    -> ProtocolV3) wires every kwarg AND runs _pcr_single_mutant_map -- pass
    cross_packaging_rate, hallucination(_rate), pcr_mutations/_rate all by keyword. NGS
    reads are length d0_ext (see ProtocolWithPCRMutations).
    """

    def _ngs_and_deplete(self, _lambda) -> tuple[jax.Array, jax.Array]:
        return _run_ngs_pipeline(
            self, _lambda, LambdaWithHallucinationAndPCRMutations,
            hallucination=self.hallucination, hallucination_rate=self.hallucination_rate,
            **self._pcr_lambda_kwargs(),
        )


