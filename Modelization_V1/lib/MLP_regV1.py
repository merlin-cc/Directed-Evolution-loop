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

import jax
from typing import Any, Callable, Sequence
from jax import random, numpy as jnp
import flax
from flax import linen as nn
import optax
import sequence_classesV1 as sc
import RegressionV1 as rg


message = "version 1.0"


