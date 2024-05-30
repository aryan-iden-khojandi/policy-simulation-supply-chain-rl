import copy
from datetime import date
import math
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from synthetic_data import *

import mpu


def compute_distance(origin_zip: str, destination_zip: str, zip_lat_longs: Dict[str, Tuple[float, float]]) -> float:
    return mpu.haversine_distance(zip_lat_longs[origin_zip], zip_lat_longs[destination_zip]) #kilometers
    # return math.dist(zip_lat_longs[origin_zip], zip_lat_longs[destination_zip])


def fulfill(
    order: Dict[str, Any],
    inventory: Dict[str, int],
    capacity: Dict[date, Dict[str, int]],
    eligible_nodes: List[Dict[str, Any]],
    zip_lat_longs: Dict[str, Tuple[float, float]]
) -> Tuple[Union[str, Dict[str, Any]], float]:
    inv_feasible: List[str] = [node for node in inventory
                               if inventory[node] > 0]

    date = order['timestamp'].date()

    cap_feasible: List[Any] = [node for node in capacity[date]
                    if capacity[date][node] > 0]

    eligible_node_ids: List[str] = [
        node['node_id'] for node in eligible_nodes]

    feasible_nodes = list(set(inv_feasible) & set(cap_feasible) & set(eligible_node_ids))

    feasible_nodes = [node for node in eligible_nodes
                      if node['node_id'] in feasible_nodes]

    if len(feasible_nodes) == 0:
        cheapest_node = 'unfulfilled'
        return cheapest_node, np.inf

    order_zip = order['destination_zip']

    distances = {
        node['node_id']: compute_distance(node['zip_code'], order_zip, zip_lat_longs)
        for node in feasible_nodes
    }

    cheapest_node = min(distances, key=distances.get)

    return cheapest_node, distances[cheapest_node]


def read_data_files() -> Tuple[
    Dict[str, Dict[str, int]],
    Dict[Any, Dict[str, int]],
    List[Dict[str, Any]],
    List[Dict[str, Any]]
]:
    with open(os.path.join(GENERATED_INPUT_DATA_DIR, INVENTORY_FILENAME), 'rb') as f:
        inventory = cPickle.load(f)

    with open(os.path.join(GENERATED_INPUT_DATA_DIR, CAPACITY_FILENAME), 'rb') as f:
        capacity = cPickle.load(f)

    with open(os.path.join(GENERATED_INPUT_DATA_DIR, ORDERS_FILENAME), 'rb') as f:
        orders = cPickle.load(f)

    with open(os.path.join(GENERATED_INPUT_DATA_DIR, NODES_FILENAME), 'rb') as f:
        nodes = cPickle.load(f)
    return inventory, capacity, orders, nodes


def simulate(
    ix_reset: int,
    product: str,
    inventory: Dict[str, int],
    capacity: Dict[date, Dict[str, int]],
    orders: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
    zip_lat_longs: Dict[str, Tuple[float, float]]
) -> Tuple[bool, int]:
    conflict = False
    conflict_ix = len(orders)

    eligible_nodes: List[Dict[str, Any]] = [
        node for node in nodes
        if product in node['inventory_dict_original']]

    for order in orders[ix_reset:]:
        date = order['timestamp'].date()

        if order['product'] == product:
            prev_fulfill = order['fulfill_from']

            # Compute new fulfillment decision
            node, __ = fulfill(order, inventory, capacity, eligible_nodes, zip_lat_longs)
            order['fulfill_from'] = node

            # Check if there is a conflict, i.e., the action
            # changes since the previous iteration
            if prev_fulfill != order['fulfill_from'] and conflict is False:
                conflict_ix = orders.index(order)
                conflict = True

            if node != 'unfulfilled':
                inventory[node] -= 1
                capacity[date][node] -= 1

        else:
            node = order['fulfill_from']

            if node != 'unfulfilled':
                capacity[date][node] -= 1

    return conflict, conflict_ix


def update_state(
    start_index: int,
    end_index: int,
    inventory: Dict[str, Dict[str, int]],
    capacity: Dict[Any, Dict[str, int]],
    orders: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]]
) -> None:
    for order in orders[start_index:end_index]:
        product = order['product']
        date = order['timestamp'].date()
        node = order['fulfill_from']

        if node != 'unfulfilled':
            inventory[product][node] -= 1
            capacity[date][node] -= 1


def initialize(directory: str, sample_path_index: int) -> Dict[str, Any]:
    results = load_directory(directory, sample_path_index)
    _, _, _, _, zip_lat_longs = read_csv_populations(POPULATION_DATA_FILE)
    results["zip_lat_longs"] = zip_lat_longs
    return results


def algorithm() -> Tuple[List[Dict[str, Any]], int]:
    inventory, capacity, orders, nodes, zip_lat_longs = initialize()

    update_from = 0
    last_conflict = 0
    k = 0

    while (True):
        k += 1
        ix_resets = []
        conflicts = []

        for product in inventory.keys():
            inv_copy = copy.deepcopy(inventory[product])
            cap_copy = copy.deepcopy(capacity)

            conflict, ix_reset = simulate(last_conflict, product, inv_copy, cap_copy, orders, nodes, zip_lat_longs)
            ix_resets.append(ix_reset)
            conflicts.append(conflict)

        if sum(conflicts) == 0:
            update_to = len(orders)
            update_state(update_from, update_to, inventory, capacity, orders, nodes)
            #print(f"ended at iteration_{k}")
            break

        else:
            last_conflict = min(ix_resets)
            update_state(update_from, last_conflict, inventory, capacity, orders, nodes)
            update_from = last_conflict
            print(f"iteration_{k}, ix_reset_{last_conflict}")
    return orders, k


def naive_algorithm() -> List[Dict[str, Any]]:
    inventory, capacity, orders, nodes, zip_lat_longs = initialize()

    for order in orders:
        date = order['timestamp'].date()
        product = order['product']
        eligible_nodes = [node for node in nodes
                          if product in node['inventory_dict_original']]

        node, __ = fulfill(order, inventory[product], capacity, eligible_nodes, zip_lat_longs)

        order['fulfill_from'] = node

        if node != 'unfulfilled':
            inventory[product][node] -= 1
            capacity[date][node] -= 1

    return orders


if __name__ == "__main__":
    #os.makedirs(GENERATED_INPUT_DATA_DIR, exist_ok=True)
    #start = time.time()
    #generate_synthetic_date()
    #print("--------")
    #print(f"Data generated in {time.time() - start} sec")

    start = time.time()
    orders, k = algorithm()
    print("--------")
    print(f"Our algorithm ended in {k} iterations after {time.time()-start} sec")
    print("--------")
    start = time.time()
    orders_naiv = naive_algorithm()
    print(f"Naive algorithm ended in {time.time() - start} sec")
    print("--------")
    print(f"Check result of both matches: {orders == orders_naiv}")
    print("--------")
