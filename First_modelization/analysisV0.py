import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from First_modelization.sequence_classes import protocol, calculate_scores, T_sel, T_viab

def pearson(a, b):
    """
    corrcoef(a,b) = [[corr(a,a)  corr(a,b)]  =  [[1  r]
                    [corr(b,a)  corr(b,b)]]     [r  1]]
    """
    a, b = np.array(a), np.array(b)
    return float(np.corrcoef(a, b)[0, 1])

def precision_at_k(score_gt, score_hat, k_frac=0.10):
    """Fraction of top-k% by GT that are also in top-k% by predicted score."""
    N   = len(score_gt)
    k   = int(N * k_frac)
    top_gt  = set(np.argsort(-np.array(score_gt))[:k])
    top_hat = set(np.argsort(-np.array(score_hat))[:k])
    return len(top_gt & top_hat) / k
