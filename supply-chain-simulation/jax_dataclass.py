from jax_dataclasses import pytree_dataclass as dataclass
from jax import lax, Array
from jaxtyping import Float, Integer, Bool
from typing import Dict, Tuple, Optional, List, Any
import jax.numpy as jnp
from jax_sim import WorkerState, AlgoState, relu
import jax

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
    distances: Float[Array, "nodes"]
    original_product_id: int
    rewards: Float[Array, "nodes"]

@dataclass
class Window1DWorkerEvent:
    original_product_id: int
    quantity: int
    node_index_near_to_far: Integer[Array, "nodes"]
    capacity_delta_other_threads: Integer[Array, "nodes"]
    distances: Float[Array, "nodes"]
    rewards: Float[Array, "nodes"]


@dataclass
class NeuralNet:
    # inventory_dual_vars: Integer[Array, "nodes-product"]
    capacity_dual_vars: Integer[Array, "nodes"]
    #original_inventory: Integer[Array, "product-nodes"]
    #original_capacity: Integer[Array, "nodes"]
    linear1_weights: Any
    linear1_biases: Any
    linear2_weights: Any
    linear2_biases: Any
    linear3_weights: Any
    linear3_biases: Any

def neural_net_initialization(layer_sizes, num_product, num_nodes, seed, FLOAT_DTYPE):
    capacity_dual_vars = jnp.zeros(num_nodes, dtype=FLOAT_DTYPE)
    ### random initialization for weights and biases
    key = jax.random.PRNGKey(seed)
    linear1_weights = jax.random.normal(key, (num_product, layer_sizes[0]), dtype=FLOAT_DTYPE) / jnp.sqrt(num_product)
    linear1_biases = jnp.zeros(layer_sizes[0], dtype=FLOAT_DTYPE)
    key = jax.random.split(key)[0]
    linear2_weights = jax.random.normal(key, (layer_sizes[0], layer_sizes[1]), dtype=FLOAT_DTYPE) / jnp.sqrt(layer_sizes[0])
    linear2_biases = jnp.zeros(layer_sizes[1], dtype=FLOAT_DTYPE)
    key = jax.random.split(key)[0]
    linear3_weights = jax.random.normal(key, (layer_sizes[1], num_nodes), dtype=FLOAT_DTYPE) / jnp.sqrt(layer_sizes[1])
    linear3_biases = jnp.zeros(num_nodes, dtype=FLOAT_DTYPE)
    return NeuralNet(capacity_dual_vars,
                        linear1_weights, linear1_biases,
                        linear2_weights, linear2_biases,
                        linear3_weights, linear3_biases)


def inventory_dual(params: NeuralNet, state: WorkerState, event: Event) -> jnp.ndarray:
 
    #x = jnp.zeros((params.original_inventory.shape[0], 1), dtype=np.float32).flatten()
    #x = x.at[event.original_product_id].set(1)
    # x = params.original_inventory[event.product][:-1] 
    # x = jnp.concatenate((x, state.inventory[event.product][:-1]), axis=0) #concatenate original and current inventory

    x = params.linear1_weights[event.original_product_id, :] + params.linear1_biases
    # Layer 1
    #x = jnp.dot(x, params.linear1_weights) + params.linear1_biases
    x = relu(x)
    
    # # Layer 2
    x = jnp.dot(x, params.linear2_weights) + params.linear2_biases
    x = relu(x)
    
    # # Layer 3
    x = jnp.dot(x, params.linear3_weights) + params.linear3_biases
    
    # Clamp output not to exceed 20
    x = jnp.clip(x, -jnp.inf, 20)
    x = jnp.exp(x)

    return x

def neural_compute(params: NeuralNet, state: WorkerState, event: Event) -> jnp.ndarray:


    x = state.inventory[:-1] #30

    x = jnp.dot(x, params.linear1_weights) + params.linear1_biases #30x64

    #x = params.linear1_weights[event.original_product_id, :] + params.linear1_biases
    # Layer 1
    #x = jnp.dot(x, params.linear1_weights) + params.linear1_biases
    x = relu(x)
    
    # # Layer 2
    x = jnp.dot(x, params.linear2_weights) + params.linear2_biases #64x64
    x = relu(x)
    
    # # Layer 3
    x = jnp.dot(x, params.linear3_weights) + params.linear3_biases #64x31

    return x


# def neural_compute(params: NeuralNet, state: WorkerState, event: Event) -> jnp.ndarray:

#     params.linear1_weights state.inventory[event.product, :-1]

def capacity_dual(params: NeuralNet, state: WorkerState, event: Event) -> jnp.ndarray:

    x = params.capacity_dual_vars

    x = jnp.clip(x, -jnp.inf, 20)
    x = jnp.exp(x)
    return x    


