import pathlib
import time

import jax
import jax.numpy as jnp
import gymnax
from jax import random
import pandas as pd
from tqdm import tqdm, trange
import sacred
import orbax
import funcy as f

from flax.training import orbax_utils
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
    n_seeds = 5
    picard = {
        "max_iters": T,
        "tol": 1e-5
    }
    lam = 1.


def setup_policy(env, env_params, policy, rng, T):
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    policy_params = policy.init(rng, init_x)
    step_rngs = jax.random.split(rng, T)
    init_state = reset(env, env_params, policy, policy_params, rng)
    return policy_params, init_state, step_rngs


def run_instance(
        sequential_fn, picard_fn, batch_policy_fn, execute_transitions_fn,
        env, env_params, policy,
        params_A, params_B,
        seed, T, lam
):
    rng = jax.random.PRNGKey(seed)
    rng_A, rng_B = jax.random.split(rng, 2)

    policy_params_A, init_A, step_rngs_A = setup_policy(
        env, env_params, policy, rng_A, T
    )
    policy_params_B, init_B, step_rngs_B = setup_policy(
        env, env_params, policy, rng_B, T
    )

    params_A["params"]["critic_net"] = policy_params_A["params"]["critic_net"]
    params_B["params"]["critic_net"] = policy_params_B["params"]["critic_net"]

    # ------- Temporary hacks ------
    params_A = jax.tree_map(
        lambda x, y: x * (1 - lam) + y * lam, params_B, params_A
    )
    init_A = init_B
    step_rngs_A = step_rngs_B
    # -------

    traj_A = sequential_fn(params_A, init_A, step_rngs_A)

    # Time sequential rollout
    start_time = time.time()
    traj_B = sequential_fn(params_B, init_B, step_rngs_B)
    traj_B.action.block_until_ready()
    seq_time = time.time() - start_time

    # Time batch policy execution
    start_time = time.time()
    actions, infos = batch_policy_fn(params_B, traj_B, step_rngs_B)
    actions.block_until_ready()
    batch_time = time.time() - start_time

    # Time transition execution
    start_time = time.time()
    x = execute_transitions_fn(traj_B.action, init_B, traj_B.policy_info, step_rngs_B)
    x.action.block_until_ready()
    transition_time = time.time() - start_time

    # Time Picard iteration
    start_time = time.time()
    traj_Bp, num_iters = picard_fn(params_B, init_B, step_rngs_B, traj_A)
    traj_Bp.action.block_until_ready()
    picard_time = time.time() - start_time

    traj_Bp2 = picard_rollout(
        env, env_params, policy,
        params_B, init_B, step_rngs_B, traj_A,
        max_iters=19, tol=1e-9
    )
    breakpoint()
    # traj_Bpp, num_iters = picard_fn(
    #     params_B, init_B, step_rngs_B, traj_Bp
    # )

    # traj_Bppp, num_iters = picard_fn(params_B, init_B, step_rngs_B, traj_Bpp)

    return_error = jnp.sum(traj_B.reward - traj_Bp.reward)
    return_ = traj_B.reward.sum()
    obs_error = (
        jnp.sqrt(jnp.sum((traj_B.obs - traj_Bp.obs) ** 2, axis=1).mean())
    )
    obs_diff = (
        jnp.sqrt(jnp.sum((traj_B.obs - traj_A.obs) ** 2, axis=1).mean())
    )
    action_error = (
        jnp.sqrt(jnp.sum((traj_B.action - traj_Bp.action) ** 2).mean())
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
        "obs_diff_AB": obs_diff,
        "obs_norm": jnp.sqrt(jnp.sum(traj_B.obs ** 2, axis=1).mean()),
        "action_error": action_error,
        "action_norm": jnp.sqrt(jnp.sum(traj_B.action ** 2).mean())
    }
    breakpoint()

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
        env, policy, picard, T, lam,
        n_seeds, _seed, output, _run,
        checkpoint_dir
):
    env_config = env
    env, env_params = make_env(**env)
    policy = make_policy(env, env_params, **policy)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=2, create=True
    )
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        checkpoint_dir, orbax_checkpointer, options
    )
    steps = checkpoint_manager.all_steps()
    #
    #
    #steps = [1651, 1652]
    checkpoint_A = checkpoint_manager.restore(steps[0])
    checkpoint_B = checkpoint_manager.restore(steps[1])
    params_A = checkpoint_A[0]["params"]
    params_B = checkpoint_B[0]["params"]

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
            env, env_params, policy,
            params_A, params_B,
            seed, T=T, lam=lam
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
