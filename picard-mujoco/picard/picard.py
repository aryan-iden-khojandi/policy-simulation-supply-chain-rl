from functools import partial
from typing import Dict, Tuple, Callable, Any

from flax import linen as nn
import gymnax
from gymnax.environments import environment
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Integer, Float

from picard.utils import *
from picard.sequential import *
from picard.nn import Policy


def replace_init_state(
        traj: RolloutStep,
        obs: Float[Array, "o_dim"],
        state: Float[Array, "s_dim"],
) -> RolloutStep:
    """
    Takes a trajectory and replaces the initial obs / state
    with the provided obs / state.

    Used for Picard iteration, where we need to force the initial
    guess trajectory to have the correct initial state.
    """
    return RolloutStep(
        obs=jnp.concatenate((jnp.expand_dims(obs, 0), traj.obs[1:])),
        state=jax.tree.map(
            lambda a, b: jnp.concatenate(
                (jnp.expand_dims(a, 0), b[1:])
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


def picard_iterator(
        env: environment.Environment,
        env_params: gymnax.EnvParams,
        policy: Policy,
        policy_params: Dict,
        init: RolloutStep,
        rngs: Integer[Array, "T 2"],
) -> Callable[[RolloutStep], RolloutStep]:
    batch_policy = jax.vmap(get_action, in_axes=(None, None, 0, 0))
    def iterator(traj: RolloutStep):
        actions, infos = batch_policy(policy, policy_params, traj, rngs)
        return execute_transitions(
            env, env_params, actions, init, infos, rngs
        )

    return iterator


def sequential_picard_iterator(
        env: environment.Environment,
        env_params: gymnax.EnvParams,
        policy: Policy,
        policy_params: Dict,
        init: RolloutStep,
        rngs: Integer[Array, "T 2"],
) -> Callable[[RolloutStep], RolloutStep]:
    def iterator(traj: RolloutStep):
        actions, infos = seq_policy(policy, policy_params, traj, rngs)
        return execute_transitions(
            env, env_params, actions, init, infos, rngs
        )
    return iterator


def seq_policy(policy, params, traj, rngs):
    return jax.lax.scan(
        lambda _, x: (None, get_action(policy, params, x[0], x[1])),
        None,
        (traj, rngs),
    )[1]


def execute_transitions(
        env,
        env_params,
        actions: Float[Array, "T a_dim"],
        init: RolloutStep,
        infos: Array,
        rngs: Integer[Array, "T 2"],
):
    """
    Executes a trajectory starting from init, given actions / infos / rngs
    """
    def transition_scanner(step, x):
        action, info, rng = x
        return transition(env, env_params, step, action, info, rng)

    rollout = accumulate(
        transition_scanner, (actions, infos, rngs), init
    )[1]
    return shift_trajectory_back(rollout, init.obs, init.state)


def picard_rollout(
        env: environment.Environment,
        env_params: gymnax.EnvParams,
        policy: Policy,
        policy_params: Dict,
        init: RolloutStep,
        rngs: Integer[Array, "T 2"],
        traj: RolloutStep,
        max_iters: int=1000,
        tol: float=1e-6,
        return_iterates: bool=False,
) -> Tuple[RolloutStep, int]:
    """
    Picard iteration until convergence to a tolerance on
    the change in actions, or max_iters iterations.
    """
    iterator = sequential_picard_iterator(
        env, env_params, policy, policy_params, init, rngs
    )

    def is_terminal(n, prev_step, curr_step):
        return jnp.logical_or(
            n >= max_iters,
            # jnp.logical_and(
            # jnp.abs(prev_step.action - curr_step.action).mean() < tol,
            jnp.sqrt(((prev_step.obs - curr_step.obs) ** 2).sum(axis=1).mean()) < tol,
            # jnp.abs(prev_step.reward - curr_step.reward).max() < tol
            # )
        )

    init_guess = replace_init_state(traj, init.obs, init.state)
    if tol <= 0:
        # If we know that max iters is the termination condition, then run
        # scan instead of while loop
        final_iterate, iterates = iterate(iterator, init_guess, max_iters)
        if return_iterates:
            return iterates, max_iters
        else:
            return final_iterate, max_iters
    else:
        return iterate_until(iterator, init_guess, is_terminal)
