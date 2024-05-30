# Performance considerations to test:
# - test array memory layouts
# - Figure out how best to ignore orders from other products
#   (orders should probably be Integer[Array, "products orders"])
# - vmap then scan or scan then vmap?
# - Don't treat unfilled orders as a special case
import pickle
from typing import Dict, Tuple, Optional, List
from functools import partial
import time
import pandas as pd
import funcy as f

from jax_dataclasses import pytree_dataclass as dataclass
import jax
from jax.experimental import checkify
from jax import lax, Array
from jax import numpy as jnp
from jaxtyping import Float, Integer, Bool
from jax.scipy.special import logsumexp
import numpy as np
from sacred import Experiment
import os
from os import path

from datamodel import NewProblem

ex = Experiment()

FLOAT_DTYPE = jnp.bfloat16

@ex.config
def config():
    problem = dict()

    algo = dict(
        max_workers=1,
        max_products_per_worker=1000000,
        num_steps=10,
        max_iters=1e7,
        layer_sizes=[30, 64, 64, 30 + 1]
    )

    input_dir = "input_dicts/300000_orders_100000_products_30_nodes_0.8_fulfillable"
    output = None #f"execution_time_{max_workers}_max_workers.txt"
    fulfill_output = None


@ex.named_config
def naive():
    max_workers = 1
    max_products_per_worker = None
    num_steps = None
    max_iters = 1
    layer_sizes = [30, 1024, 1024, 31]
    input_dir = None


# Logic for single product / trajectory
# -------------------------------------------------------------------------
@dataclass
class NeuralNet:
    params: Integer[Array, "dimension nn"]


@dataclass
class Event:
    """
    Used to represent an order for now, but could also
    represent replenishment events later on.

    worker: index of the worker to which this event should
            beassigned
    product: index of the product to which this event corresponds,
             within that worker
    """
    worker: int
    product: int
    quantity: int
    node_index_near_to_far: Integer[Array, "nodes"]
    capacity_delta_other_threads: Integer[Array, "nodes"]


@dataclass
class WorkerState:
    """
    State internal to each thread, maintained over a single trajectory.
    """
    inventory: Integer[Array, "products nodes"]
    capacity: Integer[Array, "nodes"]
    key: jax.random.PRNGKey


@dataclass
class AlgoState:
    """
    State maintained by the overall algorithm.
    """
    iteration: int
    t_reset  : int
    n_conflicts: int
    capacity : Float[Array, "nodes"]  # capacity state maintained by each product
    inventory: Float[Array, "products nodes"]  # inventory state maintained by each product
    fulfill  : Float[Array, "orders"]
    key      : jax.random.PRNGKey


# Neural net
# -------------------------------------------------------------------------
# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = jax.random.split(key)
    return(
        scale * jax.random.normal(w_key, (n, m), dtype=FLOAT_DTYPE),
        scale * jax.random.normal(b_key, (n, ), dtype=FLOAT_DTYPE)
    )

def init_network_params(sizes, key):
    """
    Initialize all layers for a fully-connected neural network
    with sizes "sizes"
    """
    keys = jax.random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k)
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def relu(x):
    return jnp.maximum(FLOAT_DTYPE(0), x)


def linear(
        x: Float[Array, "B nin"],
        p: Tuple[Float[Array, "nout nin"], Float[Array, "nout"]],
) -> Float[Array, "B nout"]:
    w, b = p
    b_broadcasted = b[:, jnp.newaxis]  # Add a new axis along the second dimension

    # Now, b_broadcasted has shape (64, 1)
    # We want to broadcast it to shape (64, batch_size)
    # We can achieve this by using JAX's broadcasting feature
    b_broadcasted = jnp.broadcast_to(b_broadcasted, (b.shape[0], x.shape[1]))

    return w @ x + b_broadcasted


def linear_relu(x, p):
    return relu(linear(x, p))


def predict(params, x):
    for param in params[:-1]:
        x = linear_relu(x, param)
    logits = linear(x, params[-1])
    return logits - logsumexp(logits)


# SIMULATOR
# -------------------------------------------------------------------------
def fulfillment(
        key: jax.random.PRNGKey,
        nn: NeuralNet,
        state: WorkerState,
        event: Event,
) -> int:
    output = predict(nn.params, state.inventory[event.product, :-1].astype(FLOAT_DTYPE))
    # TODO we need to append output here
    # node_index_near_to_far = event.node_index_near_to_far
    node_index_near_to_far = event.node_index_near_to_far
    #jax.debug.print("{i}, {j}, {k}", i=node_index_near_to_far, j = state.inventory, k=output)
    capacity = state.capacity[node_index_near_to_far]
    inventory = state.inventory[event.product][node_index_near_to_far] \
        + jnp.round(1e-7 * output / jnp.linalg.norm(output))
    # Create a mask where both conditions are met
    # inventory = state.inventory[event.product][node_index_near_to_far]
    mask = (capacity >= event.quantity) & (inventory >= event.quantity)
    index = first_nonzero(mask, -1) # take the first non-zero index (TODO need to check for feas?)
    # index = jnp.argmax(mask)

#     if force_nearest:
#         return node_index_near_to_far[index]
#     else:

    return node_index_near_to_far[index]

    # num_nodes = len(node_index_near_to_far)
    # min_cap, max_cap = jnp.min(capacity), jnp.max(capacity)
    #
    # # Heuristic: The node with greatest capacity can improve its rank by up to num_nodes/4
    # capacity_discount_factor = (num_nodes * 0.0) / (max_cap - min_cap)
    #
    # # capacity_discount_factor = 0.0  # FOR TESTING PURPOSES ONLY!!
    #
    # sample_distance_vals = jnp.flip(num_nodes - jnp.arange(num_nodes) - 1)
    # distances_by_node = jnp.arange(num_nodes)
    # distances_by_node = distances_by_node.at[node_index_near_to_far].set(sample_distance_vals)
    #
    # scalarized_obj_by_node = distances_by_node - capacity_discount_factor * capacity
    # nodes_sorted_scalarized_obj = jnp.argsort(scalarized_obj_by_node)
    # # nodes_sorted_scalarized_obj = jnp.argsort(-1.0 * capacity)  # THIS IS PURE-CAPACITY FULFILLMENT!!
    #
    # capacity = state.capacity[nodes_sorted_scalarized_obj]
    # inventory = state.inventory[nodes_sorted_scalarized_obj]
    #
    # mask = (capacity >= event.quantity) & (inventory >= event.quantity)
    # index = first_nonzero(mask, -1)  # take the first non-zero index (TODO need to check for feas?)
    #
    # return nodes_sorted_scalarized_obj[index]


def step_in_time(
        nn: NeuralNet,
        state: WorkerState,
        event: Event,
) -> Tuple[WorkerState, int]:
    init_capacity = state.capacity - event.capacity_delta_other_threads
    state = WorkerState(state.inventory, init_capacity, state.key)
    key, subkey = jax.random.split(state.key)
    node = fulfillment(subkey, nn, state, event)
    new_inventory = \
        state.inventory.at[event.product, node].add(-event.quantity)
    new_capacity = state.capacity.at[node].add(-event.quantity)
    new_state = WorkerState(new_inventory, new_capacity, key)
    return new_state, node


def simulate_product(nn, state, events):
    _, fulfill = lax.scan(
        partial(step_in_time, nn),
        state,
        events
    )
    return _, fulfill


def random_assignment(
        key: jax.random.PRNGKey,
        n_products,
        n_workers,
        max_products_per_worker
):
    """
    Randomly assign products to workers, in round-robin fashion.
    """
    assert n_workers * max_products_per_worker >= n_products
    perm_products = jax.random.permutation(key, jnp.arange(n_products))

    #in_order_assigned_id is 0, 0, 0, ..., 1, 1, 1, ....
    in_order_assigned_id = (
        jnp.arange(max_products_per_worker).repeat(n_workers, axis=0)
    )

    #in_order_assigned_worker is 0, 1, 2, ... 0, 1, 2, ...
    in_order_assigned_worker = (
        jnp.arange(n_workers)
        .reshape((1, n_workers))
        .repeat(max_products_per_worker, axis=0)
        .reshape(-1)
    )

    # we map i to worker=int(perm_products[i] / max_num_products)
    product_worker_map = in_order_assigned_worker[perm_products]
    # we map i to product_ids_in_worker = perm_products[i] % m
    product_ids_within_worker = in_order_assigned_id[perm_products]
    return product_worker_map, product_ids_within_worker


def greedy_assignment(products, n_products, n_workers, max_products_per_worker):
    product_counts = jnp.bincount(products, length=n_products)
    sorted_counts = jnp.sort(product_counts)
    argsorted_counts = jnp.argsort(product_counts)
    ranks = jnp.argsort(argsorted_counts)
    cum_counts = jnp.cumsum(sorted_counts) - sorted_counts[0]
    ideal_partition_size = len(products) / n_workers
    workers_sorted = jnp.floor_divide(cum_counts, ideal_partition_size).astype(int)
    product_worker_map = workers_sorted[ranks]
    product_ids_within_worker = cum_count(len(products), product_worker_map)
    # checkify.check(
    #     jnp.all(product_ids_within_worker < max_products_per_worker),
    #     "Trying to assign more than max_products_per_worker products to a worker"
    # )
    return product_worker_map, product_ids_within_worker


def balanced_assignment(products, n_products, n_workers, max_products_per_worker):
    product_counts = jnp.bincount(products, length=n_products)
    product_worker_map = jnp.zeros((len(products), 1))
    num_orders_by_worker = jnp.zeros((n_workers, 1))
    for idx, product_id in enumerate(product_counts):
        # Find worker with the fewest orders assigned to it
        worker_to_which_to_assign = jnp.argmin(num_orders_by_worker)
        # print(f"Worker {worker_to_which_to_assign} has lowest load of {num_orders_by_worker.at[worker_to_which_to_assign]}")
        product_worker_map = product_worker_map.at[idx].set(worker_to_which_to_assign)
        # print(f"Worker {worker_to_which_to_assign} now has load {num_orders_by_worker.at[worker_to_which_to_assign]}")
        num_orders_by_worker = num_orders_by_worker.at[worker_to_which_to_assign].set(
            num_orders_by_worker[worker_to_which_to_assign] + product_counts[idx])

    product_ids_within_worker = cum_count(len(products), product_worker_map)

    return product_worker_map, product_ids_within_worker


def cum_count(numel, x):
    """
        suppose the range of x is [0, 1, ...., numel-1]
        for each i, return the number of j such that (x[j] = x[i] and j < i)
    """
    rank = jnp.argsort(jnp.argsort(x)) # rank[i] is the #(x[j] < x[i]) + #(x[j] = x[i] and j < i), the sort is stable-sort by default 
    bincounts = jnp.bincount(x, length=numel)
    cum_bincounts = jnp.append(jnp.cumsum(bincounts), 0) #appending 0 for indexing -1
    n_less = cum_bincounts[x-1] #n_less[i] is #(x[j] < x[j])
    return rank - n_less


def first_nonzero(x, fill):
    return jax.lax.cond(
        jnp.any(x),
        lambda: jnp.argmax(x),
        lambda: fill
    )


def compute_capacity_delta_other_threads(
        n_nodes: int,
        fulfill: Integer[Array, "orders"],
        quantities: Integer[Array, "orders"],
        order_2D_index: Integer[Array, "products num_steps"],
) -> Integer[Array, "products num_steps nodes"]:
    
    one_hot_fulfill = jax.nn.one_hot(fulfill, n_nodes, dtype=jnp.int32) * quantities.reshape(-1, 1)
    caps = (-jax.lax.cumsum(one_hot_fulfill, axis=0))[order_2D_index]
    cap_deltas = jnp.diff(caps, axis=1, prepend=0)
    cap_deltas_by_product = -one_hot_fulfill[order_2D_index]
    # return jnp.zeros(cap_deltas.shape)
    return - (cap_deltas - cap_deltas_by_product)


@partial(jax.jit, static_argnames=[
    'num_steps', "n_workers",
    "max_products_per_worker"
])
def iterate_algorithm(
        nn: NeuralNet,
        algo_state: AlgoState,
        events: Event,
        num_steps: int=10,
        n_workers: int=1,
        max_products_per_worker: int=100,
) -> AlgoState:
    # jax.debug.print("t_reset {t_reset}", t_reset=algo_state.t_reset)
    # Dimensions
    n_products, n_nodes = algo_state.inventory.shape
    n_events = algo_state.fulfill.shape[0]

    # 1. Construct workers x num_steps matrix of events
    # -------------------------------------------
    # Look ahead enough steps to construct a matrix of events
    eff_num_steps = min(num_steps * n_workers, n_events)
    # The slice returned will start from t0, due to the behavior of
    # jax.lax.dynamic_slice_in_dim
    t0 = jnp.stack(
        (algo_state.t_reset, n_events - eff_num_steps)
    ).min()
    def get_slice(x):
        slice = jax.lax.dynamic_slice_in_dim(x, t0, eff_num_steps, 0)
        return jnp.roll(slice, -(algo_state.t_reset - t0), axis=0)

    # Assign products to workers
    products = get_slice(events.product)
    key, subkey = jax.random.split(algo_state.key)
    product_worker_map, product_ids_within_worker = random_assignment(
        subkey, n_products, n_workers, max_products_per_worker)
    # product_worker_map, product_ids_within_worker = greedy_assignment(
    #     events.product, n_products, n_workers, max_products_per_worker
    # )
    # product_worker_map, product_ids_within_worker = balanced_assignment(
    #     events.product, n_products, n_workers, max_products_per_worker
    # )

    workers = product_worker_map[products]

    # Construct workers x events matrices
    # This matrix contains the index in the original events array
    # corresponding to each entry in the matrix
    index_within_worker = cum_count(n_workers, workers)
    index_1D = algo_state.t_reset + jnp.arange(eff_num_steps)
    order_2D_index = (
        (-jnp.ones((n_workers, num_steps), dtype=jnp.int32))
        # if cum_count > num_steps this will automatically ignore it
        .at[workers, index_within_worker]
        .set(index_1D)
    )

    valid_mask = (
        (order_2D_index >= 0) & (order_2D_index < n_events)
    )

    capacity_delta_other_threads = compute_capacity_delta_other_threads(
        n_nodes,
        fulfill=get_slice(algo_state.fulfill),
        quantities=get_slice(events.quantity),
        order_2D_index=jnp.where(valid_mask, order_2D_index - algo_state.t_reset, -1),
    )

    events_2D = Event(
        product_worker_map[events.product[order_2D_index]],
        product_ids_within_worker[events.product[order_2D_index]],
        events.quantity[order_2D_index] * valid_mask,
        events.node_index_near_to_far[order_2D_index],
        capacity_delta_other_threads
    )

    # 2. Compute new fulfillment
    # -------------------------------------------
    # Reshape P x N inventory to workers x products_per_worker x N inventory
    inventory = (
        jnp.zeros(
            (n_workers, max_products_per_worker, n_nodes),
            dtype=FLOAT_DTYPE
        )
        .at[product_worker_map, product_ids_within_worker, :]
        .set(algo_state.inventory)
    )

    state_ctor = jax.vmap(WorkerState, in_axes=(0, None, 0))
    subkeys = jax.random.split(algo_state.key, n_workers)
    product_states = state_ctor(
        inventory,
        algo_state.capacity,
        subkeys
    )

    # shape of fulfill_2D is the same as order_2D_index
    simulate_across_products_scan_outer = jax.vmap(
        partial(simulate_product),
        in_axes=(None, 0, 0)
    )
    _, fulfill_2D = simulate_across_products_scan_outer(
        nn, product_states, events_2D
    )

    new_fulfill = algo_state.fulfill.at[order_2D_index].set(
        jnp.where(valid_mask, fulfill_2D, algo_state.fulfill[order_2D_index])
    )
    conflicts = new_fulfill != algo_state.fulfill

    # 3. Compute the new reset time, which is the minimum of:
    # -------------------------------------------
    # 1. the number of events
    # 2. the first conflict
    # 3. the first non-contiguous time in the current block
    order_flat = order_2D_index.reshape(-1).sort()
    not_ctgs = (
        # Index is not not contig if difference from previous index is > 1...
        (jnp.diff(order_flat, 1, prepend=0) > 1)
        # ... and that the previous index was not -1
        & (jnp.concatenate((jnp.array([-1]), order_flat[:-1])) >= 0)
    )
    max_t_reset = order_flat[first_nonzero(not_ctgs, len(not_ctgs)) - 1] + 1
    new_t_reset = jnp.stack(
        (first_nonzero(conflicts, max_t_reset), max_t_reset)
    ).min()

    # 4. Update state between the previous and current reset points
    # -------------------------------------------
    range_events = jnp.arange(n_events)
    update_mask = (
        (range_events >= algo_state.t_reset)
        & (range_events < new_t_reset)
    ).astype(FLOAT_DTYPE) * events.quantity
    new_inventory = (
        algo_state.inventory
        .at[events.product, new_fulfill]
        .add(-update_mask)
    )
    is_conflict = (new_t_reset < max_t_reset) & (new_t_reset != algo_state.t_reset)
    new_capacity = algo_state.capacity.at[new_fulfill].add(-update_mask)

    return AlgoState(
        algo_state.iteration + 1,
        new_t_reset,
        algo_state.n_conflicts + is_conflict.astype(int),
        new_capacity, new_inventory, new_fulfill, algo_state.key
    )


def update_array(a, b0, val):
    mask = jnp.arange(a.shape[0]) >= b0
    # Use the mask to set elements to -1
    updated_a = jnp.where(mask, val, a)

    return updated_a

@partial(
    jax.jit,
    static_argnames=[
        'num_steps', "max_workers",
        "max_products_per_worker"]
)
def run_algorithm_day(
        algo_state: AlgoState,
        events: Event,
        nn: NeuralNet,
        max_iters=10,
        num_steps=100,
        max_workers: Optional[int]=None,
        max_products_per_worker: int=100,
) -> AlgoState:
    def iterator(algo_state):
        # return iterate_algorithm(
        #     nn, algo_state, events,
        #     num_steps=num_steps,
        #     n_workers=max_workers,
        #     max_products_per_worker=max_products_per_worker,
        # )

        new_algo_state = iterate_algorithm(
            nn, algo_state, events,
            num_steps=num_steps,
            n_workers=max_workers,
            max_products_per_worker=max_products_per_worker,
        )
        new_algo_state_2 = iterate_algorithm(
            nn, new_algo_state, events,
            num_steps=num_steps,
            n_workers=max_workers,
            max_products_per_worker=max_products_per_worker,
        )
        new_fulfill = update_array(new_algo_state_2.fulfill, new_algo_state_2.t_reset,
                                   new_algo_state_2.capacity.shape[0]-1)
        new_algo_state_2 = AlgoState(
            new_algo_state_2.iteration,
            new_algo_state_2.t_reset,
            new_algo_state_2.n_conflicts,
            new_algo_state_2.capacity, new_algo_state_2.inventory,
            new_fulfill, new_algo_state_2.key
        )
        return new_algo_state_2

    return jax.lax.while_loop(
        lambda state: (
            (state.iteration < max_iters)
            & (state.t_reset < events.product.shape[0])
        ),
        iterator,
        algo_state
    )


def run_algorithm(
        capacity: List[Integer[Array, "nodes"]],
        inventory: Integer[Array, "products nodes"],
        events: List[Event],
        nn: NeuralNet,
        max_iters=10,
        num_steps=100,
        max_workers: Optional[int]=None,
        max_products_per_worker: int=100,
        seed=42,
) -> Integer[Array, "orders"]:
    """
    :param capacity: capacity[date, node] is the capacity of node on date
    :param inventory: inventory[product, node] is the starting inventory of
                      product at node, on day 0. No replenishment is allowed.
    """
    n_nodes_plus1 = inventory.shape[1]

    def process_day(key, inventory, day):
        capacity, events = day
        n_events = len(events.product)
        # Refill capacity
        state = AlgoState(
            0, 0, 0,
            capacity,
            inventory,
            jnp.ones(n_events, dtype=int) * (n_nodes_plus1 - 1),
            key
        )
        start_time = time.perf_counter()
        jit_HLO = jax.jit(
            run_algorithm_day,
            static_argnames=[
                'num_steps', "max_workers",
                "max_products_per_worker"]
        ).lower(
            state, events, nn,
            num_steps=num_steps,
            max_workers=max_workers,
            max_products_per_worker=max_products_per_worker,
            max_iters=max_iters,
        )
        jit_compiled = jit_HLO.compile()
        compile_time = time.perf_counter() - start_time
        print(f"compilation took {compile_time}s")

        start_time = time.perf_counter()
        new_state = jit_compiled(state, events, nn, max_iters=max_iters)
        run_time_1 = time.perf_counter() - start_time
        iterations_1 = new_state.iteration
        fulfill_1 = new_state.fulfill
        print(f"the first run_algorithm_day took {run_time_1}s")

        start_time = time.perf_counter()
        new_state = jit_compiled(state, events, nn, max_iters=max_iters)
        run_time_2 = time.perf_counter() - start_time
        iterations_2 = new_state.iteration
        fulfill_2 = new_state.fulfill
        print(f"the second run_algorithm_day took {run_time_2}s")

        assert jnp.all(fulfill_1 == fulfill_2)

        results = {
            "compile_time": compile_time,
            "run_time_1": run_time_1,
            "run_time_2": run_time_2,
            "conflicts": new_state.n_conflicts,
            "iterations_1": iterations_1,
            "iterations_2": iterations_2
        }

        return new_state.inventory, new_state.fulfill, results

    results = []
    fulfills = []
    key = jax.random.PRNGKey(seed)
    for day_capacity, day_events in zip(capacity, events):
        key, subkey = jax.random.split(key)
        inventory, fulfill, day_results  = process_day(
            subkey, inventory, (day_capacity, day_events)
        )
        fulfills.append(fulfill)
        results.append(day_results)

    fulfills = jnp.concatenate(fulfills)
    fulfills = fulfills.at[fulfills == n_nodes_plus1 - 1].set(-1)
    return fulfills, pd.DataFrame.from_dict(results)


def load_problem(directory, canonical_num_products, prob=None):
    if prob is None:
        prob = NewProblem.load(directory, canonical_num_products=canonical_num_products)

    inventory = prob.inventory
    capacity = prob.capacity
    products = np.array(prob.order_products)
    node_index_near_to_far = prob.node_index_near_to_far

    cap_w_unfulfill = np.concatenate(
        (capacity, np.ones_like(capacity[:, [0]]) * np.inf),
        axis=1
    )
    inv_w_unfulfill = np.concatenate(
        (inventory, np.ones_like(inventory[:, [0]]) * np.inf),
        axis=1
    )
    node_index_near_to_far_w_unfulfill = np.concatenate(
        (
            node_index_near_to_far,
            np.ones_like(node_index_near_to_far[:, [0]]) * capacity.shape[1] # num of nodes
        ),
        axis=1
    )

    algo_events = [
        Event(
            jnp.array(products[prob.order_dates == i]),
            jnp.array(products[prob.order_dates == i]),
            jnp.ones_like(products[prob.order_dates == i]),
            jnp.array(node_index_near_to_far_w_unfulfill[prob.order_dates == i, :]),
            jnp.zeros_like(node_index_near_to_far_w_unfulfill[prob.order_dates == i, :])
        )
        for i in range(prob.capacity.shape[0])
    ]

    return (
        [jnp.array(cap_w_unfulfill[i, :])
         for i in range(cap_w_unfulfill.shape[0])],
        jnp.array(inv_w_unfulfill),
        algo_events
    )


@ex.automain
def main(algo, problem, input_dir, output, fulfill_output, canonical_num_products, _seed):
    params = init_network_params(
        algo["layer_sizes"],
        jax.random.PRNGKey(_seed)
    )
    nn = NeuralNet(params)
    capacity, inventory, algo_events = load_problem(input_dir, canonical_num_products=canonical_num_products)

    ### the benchmarking does not work with multiple days yet
    ### since the compilation time for different days can vary and mix up the memory layout  
    print("--- Running parallel... ---")
    start_time = time.time()
    fulfills, results = run_algorithm(
        capacity, inventory, algo_events, nn, seed=_seed,
        **f.omit(algo, ["layer_sizes"])
    )
    parallel_time = time.time() - start_time
    print("--- total time: %s seconds ---" % (parallel_time))

    if output is not None:
        results.to_csv(output, index=False)

    if fulfill_output is not None:
        with open(fulfill_output, "wb") as file:
            np.save(file, np.array(fulfills))
