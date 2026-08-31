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
"""

import jax
import jax.numpy as jnp

from sequence_classesV1 import Lambda, ProtocolV3

MAX_MECHANISTIC_CELLS = 5_000_000  # well under where a dense (n_cells, d0) occupancy would fit in GPU memory


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


