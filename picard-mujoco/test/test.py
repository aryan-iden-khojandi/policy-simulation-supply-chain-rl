from functools import partial

import jax
import jax.numpy as jnp
import gymnax
from jax import random

from picard.picard import picard_rollout
from picard.nn import NNPolicy, MLP
from picard.sequential import *
from picard.utils import iterate, iterate_until
from picard.purejaxrl.policies import ActorCriticPolicy, ContinuousActorCritic, DiscreteActorCritic


def setup_policy(env, env_params, model, policy, rng, T):
    params = model.init(rng, jnp.zeros(3))
    step_rngs = jax.random.split(rng, T)
    init_state = reset(env, env_params, policy, params, rng)
    return params, init_state, step_rngs


def execute_batch_policy(policy, policy_params, traj, rngs):
    batch_policy = jax.vmap(get_action, in_axes=(None, None, 0, 0))
    return batch_policy(policy, policy_params, traj, rngs)


def test_picard():
    T = 10
    env, env_params = gymnax.make("Pendulum-v1")
    model = ContinuousActorCritic.from_env(env, env_params)
    policy = ActorCriticPolicy(model)
    rng = jax.random.PRNGKey(0)
    rng_A, rng_B = jax.random.split(rng, 2)

    policy_params_A, init_A, step_rngs_A = setup_policy(
        env, env_params, model, policy, rng_A, T
    )
    policy_params_B, init_B, step_rngs_B = setup_policy(
        env, env_params, model, policy, rng_B, T
    )

    seq_jit = jax.jit(partial(sequential_rollout, env, env_params, policy))
    traj_A = seq_jit(policy_params_A, init_A, step_rngs_A)
    traj_B = seq_jit(policy_params_B, init_B, step_rngs_B)

    # # Test that batch policy execution is equivalent to sequential policy execution
    # batch_policy_jit = jax.jit(partial(execute_batch_policy, policy, policy_params_A))
    # actions, infos = batch_policy_jit(traj_A, step_rngs_A)
    # breakpoint()
    # assert traj_A.action == actions


    # Test that true trajectory is fixed point of stepper
    picard_jit = jax.jit(partial(picard_rollout, env, env_params, policy, max_iters=1, tol=0.))
    traj_Ap, n_iters = picard_jit(policy_params_A, init_A, step_rngs_A, traj_A)
    traj_App, n_iters = picard_jit(policy_params_A, init_A, step_rngs_A, traj_Ap)
    breakpoint()
    print(f"converged in {n_iters} iterations.")
    assert max_error_between_trees(traj_A, traj_Ap) < 1e-6


    # # Test convergence to fixed point of stepper
    picard_jit = jax.jit(partial(picard_rollout, env, env_params, policy, max_iters=T, tol=1e-6))
    traj_Bp, n_iters = picard_jit(
        policy_params_B, init_B, step_rngs_B, traj_A)
    print(f"converged in {n_iters} iterations.")
    assert max_error_between_trees(traj_B, traj_Bp) < 1e-6


    # Test variants of picard are equivalent
    # picard_jit = jax.jit(partial(picard_rollout, env, env_params, policy, max_iters=5, tol=1e-9))


def test_iterate_until():
    f = lambda x: x + 1
    x = 0

    cond = lambda t, x, fx: jnp.logical_or(t > 1000, fx > 10)
    x_final, _ = iterate_until(f, x, cond)
    assert int(x_final) == 11

    cond = lambda t, x, fx: jnp.logical_or(t >= 3, fx > 10)
    x_trunc, _ = iterate_until(f, x, cond)
    x_trunc_2, _ = iterate(f, x, 4)
    assert int(x_trunc) == 4
    assert int(x_trunc_2) == 4


def test_iterate_and_iterate_until():
    # Test that iterate and iterate_until are equivalent
    def f(x):
        rng, y = x
        rnga, rngb = jax.random.split(rng)
        return rnga, y + jax.random.normal(rngb, y.shape)

    rng = jax.random.PRNGKey(0)
    x = (rng, jnp.zeros(3))
    x_final, iterates = iterate(f, x, 1002)
    x_final_2, _ = iterate_until(f, x, lambda t, x, fx: t > 1000)
    assert jnp.allclose(x_final[1], x_final_2[1])


def test_iterate_2():
    def f(x, rng):
        return x + jax.random.normal(rng, x.shape)

    x = jnp.zeros(3)
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 1000)
    _, x_final = accumulate(f, rngs, x)

    results = jax.vmap(lambda rng: jax.random.normal(rng, x.shape))(rngs)
    x2 = results.cumsum(axis=0)
    breakpoint()
    assert x_final == x2
