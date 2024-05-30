import time
import operator as op
from functools import partial
from typing import Tuple
import json

import funcy as f
from flax.struct import dataclass
from flax.training import orbax_utils
import orbax
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
import sacred
import wandb

from picard.purejaxrl.policies import (
    ContinuousActorCritic,
    DiscreteActorCritic,
    ActorCriticPolicy
)
from picard.purejaxrl.wrappers import (
    FlattenObservationWrapper,
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
    make_env
)
from picard.nn import Policy
from picard.picard import picard_rollout
from picard.sequential import RolloutStep, reset, sequential_rollout

ex = sacred.Experiment()

@ex.config
def default_config():
    LR = 3e-4
    NUM_ENVS = 1
    NUM_STEPS = 2048
    TOTAL_TIMESTEPS = 1e6
    UPDATE_EPOCHS = 10
    NUM_MINIBATCHES = 32
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    ENT_COEF = 0.0
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    ACTIVATION = "tanh"
    ANNEAL_LR = False
    NORMALIZE_ENV = True
    DEBUG = True
    restore = False

    SIMULATOR = {
        "algo": "sequential",
    }

    WANDB = {
        "entity": "atzheng-wandb",
        "project": "picard_neurips"
    }

# @ex.config
# def hopper():
#     env = {
#         "name": "hopper",
#         "source": "brax",
#         "backend": "spring"
#     }


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

    @classmethod
    def from_rolloutstep(cls, rolloutstep: RolloutStep):
        "Convert between Transition and RolloutStep."
        # Move "env" axis to the end
        step_t = jax.tree.map(
            lambda x: jnp.moveaxis(x, 0, 1), rolloutstep
        )
        return cls(
            done=step_t.done,
            action=step_t.action,
            value=step_t.policy_info["value"],
            reward=step_t.reward,
            log_prob=step_t.policy_info["log_prob"],
            obs=step_t.obs,
            info=step_t.info,
        )


def make_train(config, ckpt_mgr, restore_step):
    config = {k: v for k, v in config.items()}
    config["LR"] = float(config["LR"])
    config["TOTAL_TIMESTEPS"] = int(float(config["TOTAL_TIMESTEPS"]))
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env, env_params = make_env(**config["env"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        def init_policy_network(class_name, **kwargs):
            policy_classes = {
                "discrete": DiscreteActorCritic,
                "continuous": ContinuousActorCritic
            }
            network = policy_classes[class_name].from_env(
                env, env_params, **kwargs
            )
            return network

        network = init_policy_network(**config["policy"])
        policy = ActorCriticPolicy(policy=network)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

        def initialize_rollout(rng):
            rng, rng_init, rng_step = jax.random.split(rng, 3)
            rng_step = jax.random.split(rng_step, config["NUM_STEPS"])
            init = reset(
                env, env_params, policy, train_state.params, rng_init
            )
            return init, sequential_rollout(
                env, env_params, policy, train_state.params, init, rng_step
            )

        env_state, init_traj = jax.vmap(
            initialize_rollout, in_axes=(0, )
        )(reset_rng)

        # simulators = {
        #     "sequential": lambda env, envp, pol, polp, init, rng: sequential_rollout,
        #     "picard_max_iters": picard_max_iters,
        #     "picard_tol": picard_tol
        # }

        # TRAIN LOOP
        def _update_step(runner_state, iteration_number):
            # COLLECT TRAJECTORIES
            # -------------------------------------------------------------
            def collect_trajectory(
                    rng, env_state, traj
            ) -> Tuple[RolloutStep, int]:
                # Split three ways to be consistent with initialization;
                # saves executing the first trajectory.
                rng, _, rng_step = jax.random.split(rng, 3)
                rng_step = jax.random.split(
                    rng_step, config["NUM_STEPS"]
                )
                algo = config["SIMULATOR"]["algo"]
                params = {
                    k: v for k, v in config["SIMULATOR"].items()
                    if k != "algo"
                }
                if algo == "sequential":
                    trajs = sequential_rollout(
                        env, env_params, policy, train_state.params,
                        env_state, rng_step, **params
                    )
                    n_iters = 1
                elif algo == "picard":
                    trajs, n_iters = picard_rollout(
                        env, env_params, policy, train_state.params,
                        env_state, rng_step, traj, **params
                    )
                else:
                    raise ValueError

                return trajs, n_iters


                # return sequential_rollout(
                #     env, env_params, policy, train_state.params, env_state, rng_step
                # )

            # AZ: Compared to purejaxrl implementation, this repeats the final
            # state at every update
            train_state, env_state, rng, prev_trajs = runner_state
            rng, rng_step = jax.random.split(rng, 2)
            rng_step = jax.random.split(rng_step, config["NUM_ENVS"])

            trajs, n_iters = jax.vmap(collect_trajectory, in_axes=(0, 0, 0))(
                rng_step, env_state, prev_trajs
            )
            # Take the last state of the trajectory
            env_state = jax.tree.map(lambda x: x[:, -1], trajs)
            # jax.debug.print(
            #     "ts: {ts}",
            #     ts=env_state.state._env_state.timestep
            # )
            traj_batch = Transition.from_rolloutstep(trajs)

            # CALCULATE ADVANTAGE
            # ------------------------------------------------------------
            last_val = env_state.policy_info["value"]

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            
            # Debugging mode
            if config.get("DEBUG"):
                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, rng, trajs)
            metric = {
                "mean_return": jnp.sum(
                    metric["returned_episode_returns"]
                    * metric["returned_episode"]
                ) / jnp.sum(metric["returned_episode"]),
                "mean_picard_iters": jnp.mean(n_iters)
            }
            def metrics_callback(metric):
                wandb.log({k: v.tolist() for k, v in metric.items()})
            jax.debug.callback(metrics_callback, metric)

            def checkpoint_callback(x):
                n, runner_state = x
                save_args = orbax_utils.save_args_from_target(runner_state)
                ckpt_mgr.save(
                    n,
                    runner_state,
                    save_kwargs={"save_args": save_args}
                )
            jax.debug.callback(
                checkpoint_callback,
                (iteration_number, runner_state)
            )

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        # Initialize runner state
        # if restore_step:
        #     runner_state = ckpt_mgr.restore(restore_step)
        # else:
        runner_state = (train_state, env_state, _rng, init_traj)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )

        return {"runner_state": runner_state, "metrics": metric}

    return train


@ex.automain
def main(SIMULATOR, WANDB,
         output_dir,
         env, policy, restore, _config, _seed):
    rng = jax.random.PRNGKey(_seed)
    # start a new wandb run to track this script
    wandb.init(
        # track hyperparameters and run metadata
        config=f.set_in(_config, ["seed"], _seed),
        **WANDB
    )
    orbax_checkpointer = orbax.checkpoint.AsyncCheckpointer(
        orbax.checkpoint.PyTreeCheckpointHandler(), timeout_secs=50)
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=2, create=True
    )
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        output_dir, orbax_checkpointer, options)
    train_jit = jax.jit(
        make_train(
            _config,
            checkpoint_manager,
            restore_step=checkpoint_manager.latest_step() if restore else None
        )
    )

    start = time.time()
    out = train_jit(rng)
    # with open(output, "w") as file:
    #     json.dump(
    #         {
    #             "total_time": time.time() - start,
    #             "metrics": {
    #                 k: v.tolist()
    #                 for k, v in out["metrics"].items()
    #             }
    #         },
    #         file,
    #     )
    # save_args = orbax_utils.save_args_from_target(out)
