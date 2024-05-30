from typing import Any, Dict, Tuple

from flax.struct import dataclass
from flax import linen as nn
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
from jax.nn.initializers import orthogonal, constant
import numpy as np


class MLP(nn.Module):
    """Simple ReLU MLP."""

    num_hidden_units: int
    num_hidden_layers: int
    num_output_units: int
    activation: str = "tanh"
    init_scale: float = np.sqrt(2)
    final_init_scale: float = 0.01

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        for l in range(self.num_hidden_layers):
            x = nn.Dense(
                features=self.num_hidden_units,
                # kernel_init=orthogonal(self.init_scale),
                bias_init=constant(0.0)
            )(x)
            x = activation(x)

        x = nn.Dense(
            features=self.num_output_units,
            kernel_init=orthogonal(self.final_init_scale),
            bias_init=constant(0.0)
        )(x)
        return x


@dataclass
class Policy:
    """
    Abstract class for a policy. Application returns an action, and an
    arbitrary dict that may be used for learning (e.g., log probs / value fn)
    """
    def apply(
            self, obs, rng
    ) -> Tuple[Float[Array, "a_dim"], Dict[str, Any]]:
        raise NotImplementedError


@dataclass
class NNPolicy:
    """
    Policy that uses a neural network to output actions.
    """
    model: nn.Module

    def apply(
            self, params, obs, rng
    ) -> Tuple[Float[Array, "a_dim"], Dict[str, Any]]:
        action = self.model.apply(params, obs)
        return action, {}

    def init(self, *args, **kwargs):
        return self.model.init(*args, **kwargs)
