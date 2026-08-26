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

Key structural problem, either way: Protocol.produce_capsids() draws C_s ~ Poisson(rho *
lambda1(s)) INDEPENDENTLY per sequence (sequence_classesV1.py:296) -- the C_s cells
"belonging" to sequence s are disjoint from every other sequence's cells by construction.
No two different sequences ever share a simulated cell today, so there is currently no
substrate for cross-packaging to act on at all.

    ProtocolCrossPackagingMechanistic -- rebuilds a shared physical cell pool and assigns
        plasmids from ALL sequences into it, then redistributes each cell's aggregate
        capsid output across its residents. Mechanistically faithful, but O(n_cells) memory
        for cell/sequence occupancy -- n_cells = rho * N1 reaches ~1e8-1e10 in several
        sweeps already in this repo (mu_HEK_multiplicity_sweep.ipynb, diversity_sweep.ipynb
        family), which is not tractable to materialize explicitly. Included so you can see
        the shape of the problem, capped at MAX_MECHANISTIC_CELLS so it fails loudly
        instead of silently OOMing.

    ProtocolCrossPackagingBackground -- cheaper alternative: adds a population-level
        "leakage" term to produce_capsids()'s Poisson rate instead of simulating cell
        occupancy. Every sequence's rate gets += cross_packaging_rate * (mean rate among
        transfected, i.e. "healthy neighbor") -- a non-viable variant (own rate ~ 0) still
        ends up with a nonzero floor. Stays O(d0), same asymptotic cost as the existing
        produce_capsids(). This is the one I'd actually recommend starting from.
"""

import jax
import jax.numpy as jnp

from sequence_classesV1 import ProtocolV3

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


