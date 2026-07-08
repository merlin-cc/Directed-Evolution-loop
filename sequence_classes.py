import jax
import jax.numpy as jnp


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
