import time

import jax
import jax.numpy as jnp
import gymnax
from jax import random
import pandas as pd
from tqdm import tqdm, trange
import sacred

import funcy as f
from picard.picard import *
from picard.sequential import *
from picard.nn import *

from picard.purejaxrl.policies import (
    ContinuousActorCritic,
    DiscreteActorCritic,
    ActorCriticPolicy
)
from picard.purejaxrl.wrappers import make_env

ex = sacred.Experiment("rollout_times")


@ex.config
def config():
    policy = {}
    env = {}
    T = 1000
    n_seeds = 1
    param_eps = 0.1
    picard = {
        "max_iters": T,
        "tol": 1e-5
    }


def setup_policy(env, env_params, policy, rng, T):
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    policy_params = policy.init(rng, init_x)
    step_rngs = jax.random.split(rng, T)
    init_state = reset(env, env_params, policy, policy_params, rng)
    return policy_params, init_state, step_rngs


def run_instance(
        sequential_fn, picard_fn, batch_policy_fn, execute_transitions_fn,
        env, env_params, policy, seed,
        T, param_eps,
):
    rng = jax.random.PRNGKey(seed)
    rng_A, rng_B = jax.random.split(rng, 2)

    policy_params_A, init_A, step_rngs_A = setup_policy(
        env, env_params, policy, rng_A, T
    )
    policy_params_B, init_B, step_rngs_B = setup_policy(
        env, env_params, policy, rng_B, T
    )


    # ----
    # init_B = init_A
    # step_rngs_B = step_rngs_A
    # ----

    policy_params_B = jax.tree.map(
        lambda x: x + jax.random.normal(rng_B, x.shape) * param_eps,
        policy_params_A,
    )

    traj_A = sequential_fn(policy_params_A, init_A, step_rngs_A)

    # Time sequential rollout
    start_time = time.time()
    traj_B = sequential_fn(policy_params_B, init_B, step_rngs_B)
    traj_B.action.block_until_ready()
    seq_time = time.time() - start_time

    # Time batch policy execution
    start_time = time.time()
    actions, infos = batch_policy_fn(policy_params_B, traj_B, step_rngs_B)
    actions.block_until_ready()
    batch_time = time.time() - start_time

    # Time transition execution
    start_time = time.time()
    x = execute_transitions_fn(traj_B.action, init_B, traj_B.policy_info, step_rngs_B)
    x.action.block_until_ready()
    transition_time = time.time() - start_time

    # Time Picard iteration
    start_time = time.time()
    traj_Bp, num_iters = picard_fn(
        policy_params_B, init_B, step_rngs_B, traj_A
    )
    traj_Bp.action.block_until_ready()
    picard_time = time.time() - start_time

    return_error = jnp.sum(traj_B.reward - traj_Bp.reward)
    return_ = traj_B.reward.sum()
    obs_error = (
        jnp.sqrt(jnp.sum((traj_B.obs - traj_Bp.obs) ** 2, axis=1).mean())
    )
    action_error = (
        jnp.sqrt(jnp.sum((traj_B.action - traj_Bp.action) ** 2, axis=1).mean())
    )

    metrics = {
        "seq_time": seq_time,
        "picard_time": picard_time,
        "picard_iters": num_iters,
        "batch_time": batch_time,
        "transition_time": transition_time,
        "return_error": return_error,
        "return": return_,
        "obs_error": obs_error,
        "obs_norm": jnp.sqrt(jnp.sum(traj_B.obs ** 2, axis=1).mean()),
        "action_error": action_error,
        "action_norm": jnp.sqrt(jnp.sum(traj_B.action ** 2, axis=1).mean())
    }
    return metrics


def execute_batch_policy(policy, policy_params, traj, rngs):
    batch_policy = jax.vmap(get_action, in_axes=(None, None, 0, 0))
    return batch_policy(policy, policy_params, traj, rngs)


def make_policy(env, env_params, class_name, **kwargs):
    policy_classes = {
        "discrete": DiscreteActorCritic,
        "continuous": ContinuousActorCritic
    }
    network = policy_classes[class_name].from_env(
        env, env_params, **kwargs
    )
    policy = ActorCriticPolicy(policy=network)
    return policy


@ex.automain
def main(
        env, policy, picard, T, param_eps,
        n_seeds, _seed, output, _run
):
    env_config = env
    env, env_params = make_env(**env)
    policy = make_policy(env, env_params, **policy)

    seq_jit = jax.jit(
        partial(sequential_rollout, env, env_params, policy)
    )

    picard_jit = jax.jit(
        partial(picard_rollout, env, env_params, policy, **picard)
    )

    batch_policy_jit = jax.jit(
        partial(execute_batch_policy, policy)
    )

    execute_transitions_jit = jax.jit(
        partial(execute_transitions, env, env_params)
    )

    output_df = pd.DataFrame.from_dict([
        run_instance(
            seq_jit, picard_jit,
            batch_policy_jit, execute_transitions_jit,
            env, env_params, policy, seed,
            T=T, param_eps=param_eps
        )
        for seed in tqdm(range(_seed, _seed + n_seeds))
    ])

    output_df["T"] = T
    for k, v in env_config.items():
        output_df["env_" + k] = v
    for k, v in picard.items():
        output_df["picard_" + k] = v
    print(output_df)
    output_df.to_csv(output, index=False)
