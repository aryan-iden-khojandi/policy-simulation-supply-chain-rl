import csv
import concurrent
import heapq
import os
from operator import itemgetter
import random
from collections import Counter
from functools import partial
from typing import Any, Dict, List, Tuple, Union
from itertools import groupby
import multiprocessing
from operator import itemgetter
import mpu
import numpy as np
import pandas as pd
import datetime
import us
from copy import deepcopy
import _pickle as cPickle
import itertools as it
from sacred import Experiment

ex = Experiment()


DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
POPULATION_DATA_FILE = 'zip_code_information/uszips.csv'
PRODUCT_FORMAT = 'product_{}'
NODE_FORMAT = 'node_{}_{}'
GENERATED_INPUT_DATA_DIR = './input_dicts'
INVENTORY_FILENAME = 'inventory_dict_original_{sample_path_index}.p'
CAPACITY_FILENAME = 'daily_capacity_dict_{sample_path_index}.p'
ORDERS_FILENAME = 'orders_{sample_path_index}.p'
NODES_FILENAME = 'nodes_{sample_path_index}.p'


def generate_experimental_setup_specific_dir(num_orders, num_products, num_nodes, fulfillable_prop, demand=None, strict_uniform=None):
    experimental_setup_specific_dir = f'{num_orders}_orders_{num_products}_products_{num_nodes}_nodes_{fulfillable_prop}_fulfillable'
    if demand:
        for key, value in demand.items():
            experimental_setup_specific_dir += f'_demand_{key}_{value}'
    if strict_uniform:
        experimental_setup_specific_dir += '_strict_uniform'
    return experimental_setup_specific_dir

@ex.config
def config():
    problem = dict(
        num_days=1, # Number of days
        num_orders=int(1e5), # Number of orders
        num_nodes=30, # Number of nodes
        num_products=int(1e5), # Number of products
        fulfillable_prop=0.8, # Proportion of fulfillable orders
        strict_uniform=False, # Whether to generate strict uniform orders
        demand=dict(
            distribution='uniform'
        ),
        num_sample_paths=1
    )
    algo = dict()


def power_law_sampler(alpha, size):
    x = np.linspace(0, 1, size)
    y = alpha*x**(alpha-1)
    return np.random.permutation(y / np.sum(y))
    #return 1 - np.random.power(alpha, size=size)

def loglinear_sampler(beta, size):
    return np.random.permutation(np.exp(beta * np.log(np.arange(1, size+1))))

def uniform_sampler(size):
    return np.ones(size) / size

def generate_products(
        num_products: int,
        distribution: str,
        **kwargs
) -> Tuple[Dict[str, float], np.ndarray]:
    samplers = {
        'uniform': uniform_sampler,
        'power': power_law_sampler,
        'loglinear': loglinear_sampler,
        #'exponential': np.random.exponential,
        #'lognormal': np.random.lognormal
    }
    sampler = samplers[distribution]

    # Generate a set of indices for the products
    products = [PRODUCT_FORMAT.format(i)
                for i in np.arange(num_products)]

    # Generate quantities curve following a given distribution (the random sampling is done in the order generation)
    probabilities = sampler(**kwargs, size=num_products)
    quantities = probabilities / np.sum(probabilities)

    # Create a dictionary of products and their quantities
    ## quantities, i.e., probabilities are from 0 to 1
    product_quantities_dict = {
        product: quantity for product, quantity in zip(products, quantities)
    }

    return product_quantities_dict, quantities


def distribute_quantity(product_frequencies: Dict[str, float],
                        total_quantity: float) -> Dict[str, float]:
    # Calculate the total frequency
    total_frequency = sum(product_frequencies.values())

    # Initialize an empty dictionary to store the distributed quantities
    distributed_quantities = {}

    # Distribute the total quantity pro rata based on relative frequencies
    for product, frequency in product_frequencies.items():
        # Calculate the pro rata quantity based on the relative frequency
        pro_rata_quantity = total_quantity * (frequency / total_frequency)

        # Store the distributed quantity for the product
        distributed_quantities[product] = pro_rata_quantity

    return distributed_quantities


def fetch_us_state_abbreviations():
    state_abbreviations = [state.abbr for state in us.states.STATES]
    return state_abbreviations


def read_csv_populations(csv_file):
    zip_population = {}
    state_zip_population = {}
    state_population = {}
    state_most_populous_zip = {}
    canonical_us_states = fetch_us_state_abbreviations()
    zip_lat_longs = {}

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                zip_code = row['zip']
                state_id = row['state_id']
                if state_id not in canonical_us_states:
                    continue
                population = int(row['population'])
                if zip_code not in zip_lat_longs:
                    zip_lat_longs[zip_code] = (float(row['lat']), float(row['lng']))

                # Mapping zip codes to populations
                zip_population[zip_code] = population

                # Mapping state_id to zip codes to populations
                if state_id not in state_zip_population:
                    state_zip_population[state_id] = {}
                state_zip_population[state_id][zip_code] = population

                # Calculate total population for each state_id
                if state_id not in state_population:
                    state_population[state_id] = 0
                state_population[state_id] += population

                # Find the most-populous zip code for each state_id
                if state_id not in state_most_populous_zip or population > zip_population[state_most_populous_zip[state_id]]:
                    state_most_populous_zip[state_id] = zip_code
            except Exception as e:
                print(e)

    return zip_population, state_zip_population, state_population, state_most_populous_zip, zip_lat_longs


def generate_orders_date(product_frequencies, zip_populations, date, num_orders_for_date, zip_lat_longs):
    order_destinations = np.random.choice(list(zip_populations.keys()), num_orders_for_date, replace=True, p=np.array(list(zip_populations.values())) / np.sum(list(zip_populations.values())))
    order_products = np.random.choice(list(product_frequencies.keys()), num_orders_for_date, replace=True, p=np.array(list(product_frequencies.values())))
    
    # ### sort the order destinations by longitude
    # order_destinations = sorted(order_destinations, key=lambda x: -zip_lat_longs[x][1]) ## from east to west

    # ### generate an array [1, 2 ..., num_orders_for_date] and add a random number [- n /k, n / k] to each element
    # ### then sort the array and use the resulting array as the order of the orders
    # order_order = np.arange(num_orders_for_date) + np.random.uniform(-num_orders_for_date / 10, num_orders_for_date / 10, num_orders_for_date)
    # order_order = np.argsort(order_order)

    # order_destinations = [order_destinations[i] for i in order_order]


    date = datetime.datetime.combine(date, datetime.time())
    ### using numpy to generate random offsets
    offsets = np.random.randint(0, 86400-1, num_orders_for_date)
    order_timestamps = [date + datetime.timedelta(seconds=int(offset)) for offset in offsets]
    sorted_order_timestamps = sorted(order_timestamps)
    orders = [{
        'destination_zip': order_destinations[i],
        'product': order_products[i],
        'timestamp': sorted_order_timestamps[i],
        'quantity': 1
    } for i in range(num_orders_for_date)]
    return orders

def generate_orders(total_orders, product_frequencies, zip_populations, dates, zip_lat_longs):
    ### allocate orders to dates uniformly
    base = total_orders // len(dates)
    counts = [base] * len(dates)
    remaining = total_orders - base * len(dates)
    counts[-1] += remaining

    ### generate orders for each date
    orders = {
        date: generate_orders_date(product_frequencies, zip_populations, date, count, zip_lat_longs)
        for date, count in zip(dates, counts) 
    }
    return orders

def generate_total_inventory(orders: List[Dict[str, Any]],
                             fulfillable_prop: float) -> Dict[str, int]:

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(orders)

    # Group by 'product' and sum the quantities
    grouped_df = df.groupby('product')['quantity'].sum().reset_index()

    # Convert the grouped DataFrame to a dictionary
    total_inventory_by_product = dict(zip(
        grouped_df['product'],
        fulfillable_prop * grouped_df['quantity']
    ))

    return total_inventory_by_product


def generate_network_capacity(orders_by_date, fulfillable_prop):
    network_capacity = {}

    for date, order_list in orders_by_date.items():
        total_quantity = sum(order['quantity'] for order in order_list)
        capacity = int(fulfillable_prop * total_quantity)
        network_capacity[date] = capacity

    return network_capacity


def flip_dict_levels(resource_dict):
    flipped_dict = {}

    for key_outer, dict_inner in resource_dict.items():
        for key_inner, quantity in dict_inner.items():
            if key_inner not in flipped_dict:
                flipped_dict[key_inner] = {}

            flipped_dict[key_inner][key_outer] = quantity

    return flipped_dict


def generate_nodes(num_nodes, state_populations, state_zip_populations):
    # Get the state ID of the most populous state
    states_sorted_by_pop = sorted(state_populations, key=state_populations.get, reverse=True)
    states_with_dc = states_sorted_by_pop[:num_nodes]

    nodes = []

    for i, state in enumerate(states_with_dc):

        node_id = NODE_FORMAT.format(i, state)
        # Get the most-populous zip code in this state
        most_populous_zip = max(state_zip_populations[state], key=state_zip_populations[state].get)

        # Create the node dictionary
        node = {
            'node_id': node_id,
            'zip_code': most_populous_zip,
            'state': state,
            'daily_units_capacity': {},
            'inventory_dict_original': {}
        }

        nodes.append(node)

    return nodes


def prob_round(value):
    lower = np.floor(value)
    upper = np.ceil(value)
    if lower == upper:
        return lower
    else:
        return np.random.choice(
            [lower, upper],
            p=[upper - value, value - lower]
        )


# VERIFY/MODIFY
def add_resources_to_nodes(nodes, state_populations, inventory_total, network_capacity):

    node_pops = {node['node_id']: state_populations[node['state']]
                 for node in nodes}

    # Distribute inventory
    inventory_dict_original = {}
    for product in inventory_total:
        inventory_by_node = distribute_quantity(
            node_pops, inventory_total[product]
        )
        inventory_by_node = {
            node_id: prob_round(inventory_by_node[node_id])
            for node_id in inventory_by_node
        }
        ### address the case where the sum of inventory_by_node is less than inventory_total[product]
        diff = inventory_total[product] - sum(inventory_by_node.values())
        if diff > 0:
            for i in range(int(diff)):
                node_id = random.choice(list(inventory_by_node.keys()))
                inventory_by_node[node_id] += 1

        inventory_dict_original[product] = inventory_by_node

    inventory_dict_original_flipped = flip_dict_levels(inventory_dict_original)

    # Distribute capacity
    daily_capacity_by_node = {}
    for date in network_capacity:
        capacity_by_node = distribute_quantity(
            node_pops, network_capacity[date]
        )
        capacity_by_node = {
            node_id: prob_round(capacity_by_node[node_id])
            for node_id in capacity_by_node
        }
        daily_capacity_by_node[date] = capacity_by_node

    daily_capacity_by_node_flipped = flip_dict_levels(daily_capacity_by_node)

    nodes = deepcopy(nodes)
    nodes_with_resources = {}
    for node in nodes:
        node_augmented = node
        node_id = node['node_id']
        node_augmented['daily_units_capacity'] = daily_capacity_by_node_flipped[node_id]
        node_augmented['inventory_dict_original'] = inventory_dict_original_flipped[node_id]
        nodes_with_resources[node_id] = node_augmented

    return nodes, inventory_dict_original, daily_capacity_by_node


def write_data_files(
        inventory, capacity, orders, nodes,
        num_orders, num_products, num_nodes, fulfillable_prop,
        demand, strict_uniform,
        output_dir=None, sample_path_index=0
):
    experiment_specific_dir = os.path.join(
        GENERATED_INPUT_DATA_DIR,
        generate_experimental_setup_specific_dir(
            num_orders=num_orders,
            num_products=num_products,
            num_nodes=num_nodes,
            fulfillable_prop=fulfillable_prop,
            demand=demand,
            strict_uniform=strict_uniform,
        )
    )
    output_dir = output_dir or experiment_specific_dir

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, INVENTORY_FILENAME.format(sample_path_index=sample_path_index)), 'wb') as f:
        cPickle.dump(inventory, f)

    with open(os.path.join(output_dir, CAPACITY_FILENAME.format(sample_path_index=sample_path_index)), 'wb') as f:
        cPickle.dump(capacity, f)

    with open(os.path.join(output_dir, ORDERS_FILENAME.format(sample_path_index=sample_path_index)), 'wb') as f:
        cPickle.dump(orders, f)

    with open(os.path.join(output_dir, NODES_FILENAME.format(sample_path_index=sample_path_index)), 'wb') as f:
        cPickle.dump(nodes, f)


def generate_synthetic_data(
        num_products = 10, num_orders = 100000,
        fulfillable_prop = 0.8, num_nodes = 30, num_days=30,
        strict_uniform=False, output_dir=None, demand=None,
        reservation_setup=True, num_sample_paths=1
):
    demand = dict() if demand is None else demand
    # Generate products
    product_frequencies, _ = generate_products(num_products, **demand)

    # # Generate orders
    zip_populations, state_zip_population, _, _, zip_lat_longs = read_csv_populations(POPULATION_DATA_FILE)
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(num_days)]
    # dates = [dt.strftime(DATETIME_FORMAT)
    #          for dt in dates]

    for sample_path_index in range(num_sample_paths):
        orders_by_date = generate_orders(num_orders, product_frequencies, zip_populations, dates, zip_lat_longs)

        if strict_uniform: #make sure each product is ordered the same number of times
            assert num_days==1 #only works for one day
            repeat_cnt = int(num_orders / num_products)
            assert num_orders == repeat_cnt * num_products
            products = np.array(list(product_frequencies.keys()))
            products = np.repeat(products, repeat_cnt)
            products = np.random.permutation(products)
            for i, order in enumerate(orders_by_date[dates[0]]):
                order['product'] = products[i]

        # # Organize orders by date
        # orders_by_date = organize_orders_by_date(orders)

        # # Generate total inventory
        # fulfillable_prop = fulfillable_prop
        orders_flattened = [order for date in orders_by_date for order in orders_by_date[date]]
        total_inventory = generate_total_inventory(orders_flattened, fulfillable_prop)

        # Generate network capacity
        network_capacity = generate_network_capacity(orders_by_date, fulfillable_prop)

        # Generate nodes
        num_nodes = num_nodes
        state_populations = {state_id: sum(populations.values()) for state_id, populations in state_zip_population.items()}
        nodes_empty = generate_nodes(num_nodes, state_populations, state_zip_population)
        nodes, inventory_dict_original, daily_capacity_dict_original = \
            add_resources_to_nodes(nodes_empty, state_populations, total_inventory, network_capacity)

        if reservation_setup:
            assert num_nodes == 2
            # Compute the distance from each zip code to each node
            distances_to_each_node = [
                list(map(lambda x: mpu.haversine_distance(zip_lat_longs[node['zip_code']], zip_lat_longs[x]),
                                                          zip_lat_longs.keys()))
                for node in nodes
            ]

            # Find the zip codes that minimize and maximize difference in distances
            distance_diffs = np.array([distances_to_each_node[0][i] - distances_to_each_node[1][i]
                                       for i in range(len(distances_to_each_node[0]))])

            neutral_zip_idx, opportunity_zip_idx = np.argmin(distance_diffs[distance_diffs>0]), np.argmax(distance_diffs)
            zips = list(zip_lat_longs.keys())
            neutral_zip, opportunity_zip = zips[neutral_zip_idx], zips[opportunity_zip_idx]

            dists_neutral_zip = (distances_to_each_node[0][neutral_zip_idx],
                                 distances_to_each_node[1][neutral_zip_idx])
            dists_opportunity_zip = (distances_to_each_node[0][opportunity_zip_idx],
                                     distances_to_each_node[1][opportunity_zip_idx])
            print(f"Chose {neutral_zip} for the 'neutral' zip that is almost equally well-met by the two nodes: {dists_neutral_zip}; "
                  f"and {opportunity_zip} for the zip that is much better met by Node 1 (zero-indexed): {dists_opportunity_zip}")

            new_order_zips = np.random.choice([neutral_zip, opportunity_zip], size=num_orders)
            for j, order in enumerate(orders_flattened):
                order['destination_zip'] = new_order_zips[j]

        write_data_files(
            inventory_dict_original,
            daily_capacity_dict_original,
            orders_by_date,
            nodes,
            num_orders,
            num_products,
            num_nodes,
            fulfillable_prop,
            demand,
            strict_uniform,
            output_dir=output_dir,
            sample_path_index=sample_path_index
        )


def load_input_data(num_orders, num_products, num_nodes, fulfillable_prop, sample_path_index=0):

    experiment_specific_dir = os.path.join(
        GENERATED_INPUT_DATA_DIR,
        generate_experimental_setup_specific_dir(num_orders=num_orders,
                                                 num_products=num_products,
                                                 num_nodes=num_nodes,
                                                 fulfillable_prop=fulfillable_prop))
    return load_directory(experiment_specific_dir, sample_path_index)


def load_directory(directory, sample_path_index=0):

    with open(os.path.join(directory, INVENTORY_FILENAME.format(sample_path_index=sample_path_index)), 'rb') as f:
        inventory_dict_original = cPickle.load(f)

    with open(os.path.join(directory, CAPACITY_FILENAME.format(sample_path_index=sample_path_index)), 'rb') as f:
        capacity_dict_original = cPickle.load(f)

    with open(os.path.join(directory, ORDERS_FILENAME.format(sample_path_index=sample_path_index)), 'rb') as f:
        orders_by_date = cPickle.load(f)

    with open(os.path.join(directory, NODES_FILENAME.format(sample_path_index=sample_path_index)), 'rb') as f:
        nodes = cPickle.load(f)

    # orders_sorted = list(sorted(
    #     it.chain.from_iterable(orders_by_date.values()),
    #     key=itemgetter('timestamp')
    # ))
    orders_sorted = list(
        it.chain.from_iterable([
            orders_by_date[k] for k in sorted(orders_by_date.keys())
        ])
    )
    return {
        "orders": orders_sorted,
        "inventory": inventory_dict_original,
        "capacity": capacity_dict_original,
        "nodes": nodes
    }


@ex.automain
def main(problem, algo, output=None, _seed=42):
    #set random seed
    random.seed(_seed)
    np.random.seed(_seed)
    generate_synthetic_data(output_dir=output, **problem)
