import jax.numpy as jnp
from jax import lax, Array
from jaxtyping import Float, Integer, Bool
import jax 

from jax_dataclasses import pytree_dataclass as dataclass

from jax_sim import WorkerState, AlgoState, first_nonzero, compute_capacity_delta_other_threads, random_assignment, cum_count, predict, init_network_params, relu
from datamodel import NewProblem

import numpy as np
import pandas as pd
import funcy as f
from functools import partial

from typing import Dict, Tuple, Optional, List, Any
from jax_dataclass import Event, Window1DWorkerEvent, NeuralNet, neural_net_initialization, inventory_dual, capacity_dual, neural_compute

import time
FLOAT_DTYPE = jnp.bfloat16
#FLOAT_DTYPE = jnp.float32
def load_problem(directory=None, prob=None, canonical_num_products=None):
    if prob is None:
        prob = NewProblem.load(directory, canonical_num_products=canonical_num_products, sample_path_index=0)

    inventory = prob.inventory
    capacity = prob.capacity
    products = np.array(prob.order_products)
    node_index_near_to_far = prob.node_index_near_to_far
    distances = prob.distances
    
    cap_w_unfulfill = np.concatenate(
        (capacity, np.ones_like(capacity[:, [0]]) * 1e6),
        axis=1
    )
    inv_w_unfulfill = np.concatenate(
        (inventory, np.ones_like(inventory[:, [0]]) * 1e6),
        axis=1
    )
    node_index_near_to_far_w_unfulfill = np.concatenate(
        (
            node_index_near_to_far,
            np.ones_like(node_index_near_to_far[:, [0]]) * capacity.shape[1] # num of nodes
        ),
        axis=1
    )

    distances_w_unfulfill = np.concatenate(
        (distances, np.ones_like(distances[:, [0]]) * np.max(distances)),
        axis=1
    )

    algo_events = [
        Event(
            jnp.array(products[prob.order_dates == i]),
            jnp.array(products[prob.order_dates == i]),
            jnp.ones_like(products[prob.order_dates == i]),
            jnp.array(node_index_near_to_far_w_unfulfill[prob.order_dates == i, :]),
            jnp.zeros_like(node_index_near_to_far_w_unfulfill[prob.order_dates == i, :]),
            jnp.array(distances_w_unfulfill[prob.order_dates == i, :]),
            jnp.array(products[prob.order_dates == i]),
            jnp.array(-distances_w_unfulfill[prob.order_dates == i, :] / np.max(distances_w_unfulfill[prob.order_dates == i, :]))
        )
        for i in range(prob.capacity.shape[0])
    ]
    return (
        [jnp.array(cap_w_unfulfill[i, :])
         for i in range(cap_w_unfulfill.shape[0])],
        jnp.array(inv_w_unfulfill),
        algo_events
    )

def fulfillment_nearest_neighbor(
        nn: NeuralNet,
        state: WorkerState,
        event: Event,
) -> int:
    output = neural_compute(nn, state, event)
    #output = predict(nn.params, state.inventory[event.product, :-1].astype(FLOAT_DTYPE))
    # TODO we need to append output here
    # node_index_near_to_far = event.node_index_near_to_far
    node_index_near_to_far = event.node_index_near_to_far
    #jax.debug.print("{i}, {j}, {k}", i=node_index_near_to_far, j = state.inventory, k=output)
    capacity = state.capacity[node_index_near_to_far]
    inventory = state.inventory[node_index_near_to_far] \
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

def fulfillment(
        nnduals: NeuralNet,
        state: WorkerState,
        event: Event,
) -> int:
    inventory_dual_vars = inventory_dual(nnduals, state, event)
    capacity_dual_vars = capacity_dual(nnduals, state, event)
    rewards = (event.rewards - inventory_dual_vars - capacity_dual_vars)
    action = jnp.argmax(rewards)  
    return action

def fulfillment_dual_with_constraints(
        nnduals: NeuralNet,
        state: WorkerState,
        event: Event,
) -> int:
    inventory_dual_vars = inventory_dual(nnduals, state, event)
    capacity_dual_vars = capacity_dual(nnduals, state, event)
    rewards = (event.rewards - inventory_dual_vars - capacity_dual_vars)
    ### find the index with the largest reward that has both positive inventory and capacity
    ### rewards can be negative
    rewards = rewards - rewards.min() + 1 #shift rewards to be positive
    action = jnp.argmax(rewards * (state.inventory > 0) * (state.capacity > 0))
    return action

def step_in_time(
        nn: NeuralNet,
        state: WorkerState,
        event: Event,
) -> Tuple[WorkerState, int]:
    init_capacity = state.capacity - event.capacity_delta_other_threads
    node = fulfillment_nearest_neighbor(nn, WorkerState(state.inventory[event.original_product_id, :], init_capacity, state.key), event)
    new_inventory = \
        state.inventory.at[event.original_product_id, node].add(-event.quantity)
    new_capacity = state.capacity.at[node].add(-event.quantity)
    new_state = WorkerState(new_inventory, new_capacity, state.key)
    return new_state, node

def simulate_product(nn, state, events):
    _, fulfill = lax.scan(
        partial(step_in_time, nn),
        state,
        events
    )
    return _, fulfill


def derive_states_from_actions(execute_inventory, execute_capacity, actions, product_id):
    n_events = actions.shape[0]
    delta_inventory = jnp.zeros((n_events, execute_inventory.shape[1]))
    delta_inventory = delta_inventory.at[jnp.arange(n_events), actions].set(1)
    
    index = jnp.argsort(product_id) #stable sort
    inventory_cumsum = jnp.cumsum(delta_inventory[index], axis=0) ## count the cumsum of inventory for pid[j] < pid[i] or pid[j]=pid[i] and j < i
    new_inventory = delta_inventory.at[index].set(inventory_cumsum)

    bin_count_vmap = jax.vmap(lambda product_id, delta_inventory: jnp.bincount(product_id, delta_inventory, length=execute_inventory.shape[0]), in_axes=(None, 0))
    bin_count_inventory = bin_count_vmap(product_id, delta_inventory.T).T #(shape: (product_id.max()+1) x J), count the sum of inventory for pid[j]=p
    inventory_bin_cumsum = jnp.append(jnp.cumsum(bin_count_inventory, axis=0), jnp.zeros((1, bin_count_inventory.shape[1])), axis=0) # append 0 at the end so that result[-1] are all zeros

    final_inventory = execute_inventory[product_id, :] - (new_inventory - inventory_bin_cumsum[product_id-1]) + delta_inventory
    updated_execute_inventory = execute_inventory - bin_count_inventory

    delta_capacity = jnp.zeros((n_events, execute_capacity.shape[0]))
    delta_capacity = delta_capacity.at[jnp.arange(n_events), actions].set(1)
    capacity_cumsum = jnp.cumsum(delta_capacity, axis=0)

    final_capacity = execute_capacity - (capacity_cumsum - delta_capacity)
    updated_execute_capacity = execute_capacity - capacity_cumsum[-1]

    return updated_execute_inventory, updated_execute_capacity, final_inventory, final_capacity

@partial(jax.jit, static_argnames=['num_steps', 'fulfillment_func'])
def iterate_algorithm_with_1D_window_size(
    nn,  # Assuming NeuralNet is passed correctly
    algo_state,
    events,
    num_steps: int = 10,
    fulfillment_func = None
) -> AlgoState:
    
    n_products, n_nodes = algo_state.inventory.shape
    n_events = algo_state.fulfill.shape[0]
    
    # Construct WorkerState
    inventory = jnp.zeros((num_steps, n_nodes), dtype=FLOAT_DTYPE)
    capacity = jnp.zeros((num_steps, n_nodes), dtype=FLOAT_DTYPE)
    t0 = algo_state.t_reset

    execute_inventory = algo_state.inventory
    execute_capacity = algo_state.capacity

    ## Construct WorkerEvent using dynamic slicing
    product = jax.lax.dynamic_slice(events.product, (t0,), (num_steps,))
    quantity = jax.lax.dynamic_slice(events.quantity, (t0,), (num_steps,))
    node_index_near_to_far = jax.lax.dynamic_slice(events.node_index_near_to_far, (t0, n_nodes), (num_steps, n_nodes))
    capacity_delta_other_threads = jnp.zeros((num_steps, n_nodes), dtype=FLOAT_DTYPE)
    distances = jax.lax.dynamic_slice(events.distances, (t0, n_nodes), (num_steps, n_nodes))
    rewards = jax.lax.dynamic_slice(events.rewards, (t0, n_nodes), (num_steps, n_nodes))
    old_fulfill = jax.lax.dynamic_slice(algo_state.fulfill, (t0,), (num_steps,))

    event = Window1DWorkerEvent(product, quantity, node_index_near_to_far, capacity_delta_other_threads, distances, rewards)


    ## Execute the inventory and capacity update using algo.fulfill for t0 to t0+num_steps
    execute_inventory, execute_capacity, inventory, capacity = derive_states_from_actions(execute_inventory, execute_capacity, old_fulfill, product)

    ### copy algo_state.key num_steps times (not split)
    keys = jnp.repeat(algo_state.key, num_steps, axis=0).reshape(num_steps, -1)
    state = WorkerState(inventory, capacity, keys)




    ### Run fulfillment in vmap [fulfillment takes a single event and returns a single action]
    fulfill = jax.vmap(fulfillment_func, in_axes=(None, 0, 0))(nn, state, event)

    #jax.debug.print("fulfill {fulfill}", fulfill=fulfill)


    ## Check if there is a conflict
    is_conflict = jnp.any(fulfill != old_fulfill)

    ## Update algo_state
    def update_algo_state(algo_state, fulfill, execute_inventory, execute_capacity, t0, num_steps, is_conflict):
        
        # Update the fulfill attribute
        new_fulfill = jax.lax.dynamic_update_slice(algo_state.fulfill, fulfill, (t0,))
        #jax.debug.print("new_fulfill {new_fulfill}", new_fulfill=new_fulfill)
       
        # Conditional update
        def update_on_no_conflict(algo_state):
            return AlgoState(
                algo_state.iteration + 1,
                t0 + num_steps,
                algo_state.n_conflicts,
                execute_capacity,
                execute_inventory,
                new_fulfill,
                algo_state.key
            )
        
        def update_on_conflict(algo_state):
            return AlgoState(
                algo_state.iteration + 1,
                t0,
                algo_state.n_conflicts + 1,
                algo_state.capacity,
                algo_state.inventory,
                new_fulfill,
                algo_state.key
            )
        
        # Apply conditional logic
        algo_state = jax.lax.cond(
            is_conflict,
            update_on_conflict,
            update_on_no_conflict,
            algo_state
        )
        
        return algo_state
    algo_state = update_algo_state(algo_state, fulfill, execute_inventory, execute_capacity, t0, num_steps, is_conflict)
    return algo_state

@partial(
    jax.jit,
    static_argnames=[
        'num_steps', "max_workers",
        "max_products_per_worker", "fulfillment_func"]
)
def run_algorithm_day(
        algo_state: AlgoState,
        events: Event,
        nn: NeuralNet,
        max_iters=10,
        num_steps=100,
        fulfillment_func = None,
        max_workers: Optional[int]=None,
        max_products_per_worker: int=100,
) -> AlgoState:
    def iterator(algo_state):
        return iterate_algorithm_with_1D_window_size(
            nn, algo_state, events,
            num_steps=num_steps,
            fulfillment_func = fulfillment_func
        )

        # return iterate_algorithm(
        #     nn, algo_state, events,
        #     num_steps=num_steps,
        #     n_workers=max_workers,
        #     max_products_per_worker=max_products_per_worker,
        # )

    return jax.lax.while_loop(
        lambda state: (
            (state.iteration < max_iters)
            & (state.t_reset < events.product.shape[0])
        ),
        iterator,
        algo_state
    )

def run_sequential_algorithm_day(
        algo_state: AlgoState,
        events: List[Event],
        nn: NeuralNet,
        seed=42):
    state = WorkerState(algo_state.inventory, algo_state.capacity, jax.random.PRNGKey(seed))
    _, fulfill = simulate_product(nn, state, events)
    #jax.debug.print("fulfill {fulfill}", fulfill=fulfill)
    return fulfill

def jit_compile(func, *args, static_argnames=[], **kwargs):
    return jax.jit(func, static_argnames = static_argnames).lower(*args, **kwargs).compile()

import funcy as f
import time
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import optax
from jax.tree_util import tree_map


def lagrangian_dual_descent(nnduals, problem_instance_samples, num_iters = 100, lr=0.01, batch_size = 1000, _seed = 42, jax_params = None, debug_print_step = 50, mode = "parallel"):

    
    print("Preprocessing")

    algo_events_list = []
    jit_fulfill_list = []
    algo_states_list = []
    inventory_original_normalized_list = []
    capacity_original_normalized_list = []
    for prob in problem_instance_samples:
        print("Load Problem")
        capacity, inventory, algo_events = load_problem(directory=None, prob=prob)
        capacity_original = capacity[0]
        inventory_original = inventory
        algo_events = algo_events[0]
        algo_states = AlgoState(
            0, 0, 0,
            capacity_original,
            inventory_original,
            jnp.ones(algo_events.product.shape[0], dtype=int) * (len(capacity_original) - 1),
            jax.random.PRNGKey(42)
        )
    
        print("Compiling")
        start_time = time.time()
        if mode == "parallel":
            jit_fulfill = jit_compile(run_algorithm_day, algo_states, algo_events, nnduals, max_iters=jax_params["max_iters"], fulfillment_func=fulfillment_nearest_neighbor, num_steps=jax_params["num_steps"], static_argnames = ['num_steps', "max_workers", "max_products_per_worker", "fulfillment_func"])
        elif mode == "sequential":
            jit_fulfill = jax.jit(run_sequential_algorithm_day)
        print(f"Compilation Time: {time.time() - start_time}")
        products_count = jnp.bincount(algo_events.product, length=inventory_original.shape[0])
        inventory_original_normalized = inventory_original[algo_events.product, :] / products_count[algo_events.product].reshape(-1, 1)
        capacity_original_normalized = capacity_original / algo_events.product.shape[0]

        jit_fulfill_list.append(jit_fulfill)
        algo_events_list.append(algo_events)
        algo_states_list.append(algo_states)
        inventory_original_normalized_list.append(inventory_original_normalized)
        capacity_original_normalized_list.append(capacity_original_normalized)

    elapse_fulfillment_time_list = []
    elapse_gradient_time_list = []
    solver = optax.adam(learning_rate=lr)
    opt_state = solver.init(nnduals)

    def compute_loss_jax(nnduals, algo_events, states, fulfill, inventory_original_normalized, capacity_original_normalized):
        def compute_loss_jax_inner(nnduals, state, event, fulfill, inventory_original_normalized, capacity_original_normalized):
            inventory_duals = inventory_dual(nnduals, state, event)
            capacity_duals = capacity_dual(nnduals, state, event)

            delta = jnp.zeros_like(event.rewards)
            delta = delta.at[fulfill].set(1)

            loss = inventory_duals * (inventory_original_normalized - delta) + capacity_duals * (capacity_original_normalized - delta) 
            loss = loss[:-1].sum() + jnp.sum(delta * event.rewards)
            return loss
        
        loss = jax.vmap(compute_loss_jax_inner, in_axes=(None, None, 0, 0, 0, None))(nnduals, states, algo_events, fulfill, inventory_original_normalized, capacity_original_normalized)
        loss = loss.sum()
        return loss

    grad_compute_loss_jax = jax.grad(compute_loss_jax)
    key = jax.random.PRNGKey(_seed)

    for episode in range(num_iters):
        for jit_fulfill, algo_events, algo_states, inventory_original_normalized, capacity_original_normalized in zip(jit_fulfill_list, algo_events_list, algo_states_list, inventory_original_normalized_list, capacity_original_normalized_list):
            start_time = time.time()
            if mode == "parallel":
                new_state = jit_fulfill(algo_states, algo_events, nnduals, max_iters=jax_params["max_iters"])
                fulfill = new_state.fulfill
            elif mode == "sequential":
                fulfill = jit_fulfill(algo_states, algo_events, nnduals)
            fulfill = fulfill.at[fulfill == -1].set(len(capacity_original) - 1)
            ## Execute the inventory and capacity from the fulfill
            _, _, inventory, capacity = derive_states_from_actions(algo_states.inventory, algo_states.capacity, fulfill, algo_events.product)
            n_events = algo_events.product.shape[0]
            keys = jnp.repeat(key, n_events, axis=0).reshape(n_events, -1)
            states = WorkerState(inventory, capacity, keys)
            elapse_fulfillment_time_list.append(time.time() - start_time)

            start_time = time.time()
            grads = grad_compute_loss_jax(nnduals, algo_events, states, fulfill, inventory_original_normalized, capacity_original_normalized)
            updates, opt_state = solver.update(grads, opt_state, nnduals)
            nnduals = optax.apply_updates(nnduals, updates)
            elapse_gradient_time_list.append(time.time() - start_time)


            # action_correct = np.zeros((len(orders)), dtype=np.int32)

            # for i in range(0, len(rewards), batch_size):
            #     state = {"product_id": states["product_id"][i:i+batch_size], "original_inventory": states["original_inventory"][i:i+batch_size], "original_capacity": states["original_capacity"][i:i+batch_size]}
            #     action_correct[i:i+batch_size] = policy_net.action(state, rewards[i:i+batch_size])
            
            # #print(action_correct[120:130])
            # for i in range(len(action)):
            #     if action[i] != action_correct[i]:
            #         print("Error")
            #         print(i, action[i], action_correct[i])
            #         state = {"product_id": states["product_id"][i], "original_inventory": states["original_inventory"][i].unsqueeze(0), "original_capacity": states["original_capacity"][i].unsqueeze(0)}
            #         reward = rewards[i].unsqueeze(0)
            #         print(policy_net.forward(state, reward))



            # start_time = time.time()    
            # for i in range(0, len(rewards), batch_size):
            #     state = {"product_id": states["product_id"][i:i+batch_size], "original_inventory": states["original_inventory"][i:i+batch_size], "original_capacity": states["original_capacity"][i:i+batch_size]}
            #     reward = rewards[i:i+batch_size]
            #     loss = policy_net.compute_loss(state, reward, inventory_original_normalized[i:i+batch_size], capacity_original_normalized[i:i+batch_size], action[i:i+batch_size])
            #     total_reward += loss.sum()
            #     loss.sum().backward()
                
            # optimizer.step()

        if episode % debug_print_step == 0:
            print(f"Episode {episode}: Mean Fulfillment Elapse Time {np.mean(elapse_fulfillment_time_list[1:])}")
            print(f"Episode {episode}: Mean Gradient Elapse Time {np.mean(elapse_gradient_time_list[1:])}")
            print()

            
        if episode % 50 == 0:
            valid_index = 0

            algo_events = algo_events_list[valid_index]
            algo_states = algo_states_list[valid_index]
            new_state = run_algorithm_day(algo_states, algo_events, nnduals, max_iters=jax_params["max_iters"], fulfillment_func = fulfillment_dual_with_constraints, num_steps=jax_params["num_steps"])
            fulfill = new_state.fulfill
            fulfill = fulfill.at[fulfill == -1].set(algo_states.capacity.shape[0] - 1)
            valid_distance = algo_events.distances[jnp.arange(fulfill.shape[0]), fulfill].mean()
            print(f"Episode {episode}")
            print(f"Valid Distance: {valid_distance}")

            ## Execute the inventory and capacity from the fulfill
            _, _, inventory, capacity = derive_states_from_actions(algo_states.inventory, algo_states.capacity, fulfill, algo_events.product)
            n_events = algo_events.product.shape[0]
            keys = jnp.repeat(key, n_events, axis=0).reshape(n_events, -1)
            states = WorkerState(inventory, capacity, keys)
        
            total_loss = compute_loss_jax(nnduals, algo_events, states, fulfill, inventory_original_normalized, capacity_original_normalized)
            print(f"Total loss: ", total_loss)


            #print(f"Inventory Dual: {policy_net.inventory_dual_vars}")
            #print(f"Capacity Dual: {policy_net.capacity_dual_vars}")
    return nnduals
         
     


import primal_dual_fulfillment as pd_fulfillment
from sacred import Experiment

ex = Experiment("jax_E2E_RL")

@ex.config
def config():
    problem = dict()

    algo = dict(
        max_workers=1,
        max_products_per_worker=1000000,
        num_steps=10,
        max_iters=1e7,
        layer_sizes=[64, 64]
    )

    optimizer = dict(
        lr=0.005,
        batch_size=1000,
        num_iters=1000,
        debug_print_step=50,
        mode="parallel"
    )

    input_dir = "input_dicts/300000_orders_100000_products_30_nodes_0.8_fulfillable"
    output = None #f"execution_time_{max_workers}_max_workers.txt"
    fulfill_output = None
    _seed = 42

    output_greedy = False



@ex.automain
def main(algo, problem, optimizer, input_dir, canonical_num_products, _seed, output_greedy):
    params = init_network_params(
        algo["layer_sizes"],
        jax.random.PRNGKey(_seed)
    )
    prob = NewProblem.load(input_dir, canonical_num_products, 0)
    problem_instance_samples = [prob]

    if output_greedy:
        def create_orders(prob, inf_inventory = 1e6, unfulfill_distance_coef = 1):
            print('average_distance (km):', prob.distances.mean())
            max_distance = prob.distances.max()
            print('max_distance (km):', max_distance)
            orders = pd_fulfillment.generate_orders_from_prob(prob, unfulfill_distance = max_distance * unfulfill_distance_coef)
            inventory = np.append(prob.inventory, inf_inventory * np.ones((prob.inventory.shape[0], 1)), axis=1)
            capacity = np.append(prob.capacity[0], inf_inventory)
            return orders, inventory, capacity

        orders_list = []
        inventory_list = []
        capacity_list = []
        for prob in problem_instance_samples:
            orders, inventory, capacity = create_orders(prob)
            orders_list.append(orders)
            inventory_list.append(inventory)
            capacity_list.append(capacity)

        def greedy_policy_evaluation(orders, inventory, capacity):
            orders = pd_fulfillment.naive_policy(orders, inventory.copy(), capacity.copy())
            reward = pd_fulfillment.calculate_cumulative_reward(orders)
            average_distance, fufillment_rate = pd_fulfillment.calculate_cost(orders)
            print("greedy average distances",average_distance)
            print("greedy fulfillment_rate", fufillment_rate)
            #return reward, average_distance, fufillment_rate

        for i in range(len(orders_list)):
            greedy_policy_evaluation(orders_list[i], inventory_list[i], capacity_list[i])

    else:
        torch.manual_seed(_seed)
        nn_duals = neural_net_initialization(algo["layer_sizes"], num_product=30, num_nodes=prob.inventory.shape[1] + 1, seed = _seed, FLOAT_DTYPE=FLOAT_DTYPE)

        print("--- Running parallel... ---")
        start_time = time.time()
        lagrangian_dual_descent(nn_duals, problem_instance_samples[0:2], _seed = _seed, jax_params = algo, **optimizer)
        parallel_time = time.time() - start_time
        print("--- total time: %s seconds ---" % (parallel_time))