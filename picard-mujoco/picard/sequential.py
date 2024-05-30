from functools import partial
from typing import Dict, Tuple, Callable, Any, Union

from flax import linen as nn
from flax.struct import dataclass
import gymnax
from gymnax.environments import environment
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Integer, Float, Shaped, Bool

from picard.utils import *


@dataclass
class RolloutStep:
    obs: Float[Array, "o_dim"]
    state: Float[Array, "s_dim"]
    action: Union[Float[Array, "a_dim"], Integer[Array, "a_dim"]]
    reward: Float[Array, "1"]
    done: Bool[Array, "1"]
    info: Shaped[Array, "any"]
    policy_info: Dict

    @classmethod
    def from_state_and_policy(obs, state, policy, policy_params):
        """
        Creates a rolloutstep with the correct types for iterating
        with a given obs / state and policy.
        Useful for e.g. initializing jax.lax.scan.
        """
        dummy_step = RolloutStep(obs, state, None, 0., False, {}, {})
        next_step = step(env, env_params, policy, policy_params, dummy_step, rng)
        return RolloutStep(
            obs, state,
            next_step.action, next_step.reward, next_step.done,
            next_step.info, next_step.policy_info
        )


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




def transition(
        env: environment.Environment,
        env_params: gymnax.EnvParams,
        step: RolloutStep,
        action: Float[Array, "a_dim"],
        policy_info: Dict,
        rng: Integer[Array, "2"]
) -> RolloutStep:
    "Execute state transition"
    action_rng, step_rng = jax.random.split(rng, 2)
    new_obs, new_state, reward, done, info = env.step(
        step_rng, step.state, action, env_params
    )
    return RolloutStep(
        new_obs, new_state, action, reward, done, info, policy_info
    )


def get_action(
        policy: Policy,
        policy_params: Dict,
        rollout_step: RolloutStep,
        rng: Integer[Array, "2"]
) -> Tuple[Float[Array, "a_dim"], Dict]:
    "Execute policy to get action."
    action_rng, step_rng = jax.random.split(rng, 2)
    return policy.apply(policy_params, rollout_step.obs, action_rng)


def step(
        env: environment.Environment,
        env_params: gymnax.EnvParams,
        policy: Policy,
        policy_params: Dict,
        rollout_step: RolloutStep,
        rng: Integer[Array, "2"]
) -> RolloutStep:
    "Execute policy to get action, then transition the environment."
    action, policy_info = get_action(
        policy, policy_params, rollout_step, rng
    )
    return transition(
        env, env_params, rollout_step, action, policy_info, rng
    )


def reset(
        env: environment.Environment,
        env_params: gymnax.EnvParams,
        policy: Policy,
        policy_params: Dict,
        rng: Integer[Array, "2"]
) -> RolloutStep:
    "Obtain an initial RolloutStep from the environment."
    obs, state = env.reset(rng)
    # Take one actual step to ensure we have the right types for all fields
    dummy_step = RolloutStep(obs, state, None, 0., False, {}, {})
    next_step = step(env, env_params, policy, policy_params, dummy_step, rng)
    return RolloutStep(
        obs, state,
        next_step.action, next_step.reward, next_step.done,
        next_step.info, next_step.policy_info
    )


def sequential_rollout(
        env: environment.Environment,
        env_params: gymnax.EnvParams,
        policy: nn.Module,
        policy_params: Dict,
        init: RolloutStep,
        rngs: Integer[Array, "T 2"],
) -> RolloutStep:
    """
    Rollout the environment sequentially for T steps."

    step must have signature (RolloutStep, rng) -> RolloutStep
    """
    stepper = partial(step, env, env_params, policy, policy_params)
    rollout = accumulate(stepper, rngs, init)[1]
    return shift_trajectory_back(rollout, init.obs, init.state)


def shift_trajectory_back(
        traj: RolloutStep,
        obs: Float[Array, "o_dim"],
        state: Float[Array, "s_dim"],
) -> RolloutStep:
    """
    Takes a trajectory {o[t+1], s[t+1], a[t], r[t], d[t]}
    and shifts o and s back by one time step, to get
    {o[t], s[t], a[t], r[t], d[t]}
    """
    return RolloutStep(
        obs=jnp.concatenate((jnp.expand_dims(obs, 0), traj.obs[:-1])),
        state=jax.tree.map(
            lambda a, b: jnp.concatenate(
                (jnp.expand_dims(a, 0), b[:-1])
            ),
            state,
            traj.state
        ),
        action=traj.action,
        reward=traj.reward,
        done=traj.done,
        info=traj.info,
        policy_info=traj.policy_info
    )
