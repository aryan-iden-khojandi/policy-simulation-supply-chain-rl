from dataclasses import field
from typing import Sequence, Dict

import funcy as f
from flax import linen as nn
from flax.struct import dataclass
import numpy as np
import distrax
import jax.numpy as jnp
from jax.nn.initializers import orthogonal, constant
from picard.nn import Policy, MLP



class DiscreteActorCritic(nn.Module):
    """
    Copied directly from purejaxrl/ppo.py
    """
    action_dim: Sequence[int]
    activation: str = "tanh"
    actor: dict = field(default_factory=lambda: {
        "num_hidden_units":  64,
        "num_hidden_layers":  2,
        "init_scale": np.sqrt(2),
        "final_init_scale": 0.01
    })
    critic: dict = field(default_factory=lambda: {
        "num_hidden_units":  64,
        "num_hidden_layers":  2,
        "init_scale": np.sqrt(2),
        "final_init_scale": 0.01
    })

    def setup(self):
        self.actor_net= MLP(
            num_output_units=self.action_dim,
            activation=self.activation,
            **self.actor
        )

        self.critic_net= MLP(
            num_output_units=1,
            activation=self.activation,
            **self.critic
        )

    @nn.compact
    def __call__(self, x):
        actor_mean = self.actor_net(x)
        pi = distrax.Categorical(logits=actor_mean)
        critic = self.critic_net(x)
        return pi, jnp.squeeze(critic, axis=-1)

    @classmethod
    def from_env(cls, env, env_params, **kwargs):
        return cls(env.action_space(env_params).n, **kwargs)



class ContinuousActorCritic(nn.Module):
    """
    Copied directly from purejaxrl/ppo_continuous.py
    """
    action_dim: Sequence[int]
    activation: str = "tanh"
    actor: dict = field(default_factory=lambda: {
        "num_hidden_units":  256,
        "num_hidden_layers":  2,
        "init_scale": np.sqrt(2),
        "final_init_scale": 0.01
    })

    critic: dict = field(default_factory=lambda: {
        "num_hidden_units":  256,
        "num_hidden_layers":  2,
        "init_scale": np.sqrt(2),
        "final_init_scale": 1.0
    })

    def setup(self):
        self.actor_net = MLP(
            num_output_units=self.action_dim,
            activation=self.activation,
            **self.actor
        )

        self.critic_net = MLP(
            num_output_units=1,
            activation=self.activation,
            **self.critic
        )

    @nn.compact
    def __call__(self, x):
        actor_mean = self.actor_net(x)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        critic = self.critic_net(x)
        return pi, jnp.squeeze(critic, axis=-1)


    @classmethod
    def from_env(cls, env, env_params, **kwargs):
        return cls(
            env.action_space(env_params).shape[0],
            **kwargs
        )


@dataclass
class ActorCriticPolicy(Policy):
    policy: nn.Module
    perturb = 0.

    def apply(self, params, obs, rng):
        pi, value = self.policy.apply(params, obs)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        info = {
            "value": value,
            "log_prob": log_prob
        }
        return action, info

    def init(self, *args, **kwargs):
        return self.policy.init(*args, **kwargs)
