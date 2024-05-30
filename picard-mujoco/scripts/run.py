import jax
import jax.numpy as jnp
import gymnax
from jax import random
import pandas as pd
from tqdm import tqdm, trange
import time

from picard import *


def run_instance(seed, param_eps=0., different_rng=False):
    T = 200
    model = MLP(48, 2, 1)
    env, env_params = gymnax.make("Pendulum-v1")
    rng = jax.random.PRNGKey(seed)
    rng_A, rng_B = jax.random.split(rng, 2)
    step_rngs = jax.random.split(rng, T)

    policy_params_A = model.init(rng_A, jnp.zeros(3), None)
    problem_A = Problem(env, env_params, model, policy_params_A)
    traj_A = sequential_rollout(problem_A, step_rngs)

    # Same params, different rng
    problem_B = Problem(
        env, env_params, model,
        jax.tree.map(
            lambda x: x + jax.random.normal(rng_B, x.shape) * param_eps,
            policy_params_A
        )
    )
    # step_rngs_B = (
    #     jax.vmap(jax.random.split, in_axes=(0, None))
    #     (step_rngs, 1)
    #     .reshape((T, 2))
    #     # .at[0]
    #     # .set(step_rngs[0])
    # )
    step_rngs_B = jax.random.split(rng_B, T)

    traj_B = sequential_rollout(problem_B, step_rngs_B)
    # traj_Bp = picard_rollout_all_v2(problem_B, step_rngs_B, traj_A, T)


    # Compute errors on all parts of the trajectory
    mses = jax.tree.map(
        lambda a, b: (
            (
                b.astype(float)
                - jnp.expand_dims(a, 0).astype(float)
            ) ** 2
        ).mean(axis=tuple(range(1, b.ndim))),
        traj_B,
        traj_Bp
    )

    output = pd.DataFrame(
        {
            "t": jnp.arange(T).astype(int),
            "obs": mses.obs,
            "action": mses.action,
            "reward": mses.reward,
            "seed": seed
        }
    )
    return output


output = pd.concat([
    run_instance(seed) for seed in tqdm(range(10))
])
output.to_csv("mses.csv", index=False)
