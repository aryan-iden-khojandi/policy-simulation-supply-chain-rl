#!/usr/bin/env python3
import datetime
from datetime import date
from dataclasses import dataclass
from typing import Dict, List, Tuple

import funcy as f
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

import fulfillment_optimization as fo


@dataclass
class Order:
    product        : int
    destination_zip: str
    timestamp      : datetime.datetime
    quantity       : int  # NB: Other parts of the code (such as step_inventory_capacity()) still assume quantity=1


@dataclass
class Node:
    node_id : str
    zip_code: str
    state   : str
    daily_units_capacity: Dict
    inventory_dict_original: Dict


@dataclass
class State:
    t        : int  # index up to which inventory and capacity have been computed (non-inclusive)
    inventory: NDArray[np.int_]  # prods x nodes
    capacity : Dict[date, NDArray[np.int_]]  # orders x nodes
    fulfill  : np.ndarray


@dataclass
class Problem:
    inventory             : Dict[str, Dict[str, int]]
    capacity              : Dict[date, np.ndarray]
    orders                : List[Order]
    nodes                 : List[Node]
    distances             : np.ndarray  # orders x nodes
    node_index_near_to_far: np.ndarray  # orders x nodes
    init_inventory        : np.ndarray
    num_products          : int

    @classmethod
    def load(cls):
        inventory, capacity, orders, nodes, zip_lat_longs = fo.initialize()
        product_id_to_int = {
            product: i for i, product in enumerate(inventory.keys())
        }
        node_id_to_int = {
            node['node_id']: i for i, node in enumerate(nodes)
        }

        # Convert dicts into objects, and convert str ids to ints
        orders = [
            Order(**f.merge(
                f.omit(order, "fulfill_from"),
                dict(product=product_id_to_int[order["product"]])))
            for order in orders
        ]

        nodes = [
            Node(**f.merge(
                node,
                dict(node_id=node_id_to_int[node["node_id"]])))
            for node in nodes
        ]

        capacity = {
            date: {
                node_id_to_int[node_id]: capacity
                for node_id, capacity in node_capacities.items()
            }
            for date, node_capacities in capacity.items()
        }

        inventory = {
            product: {
                node_id_to_int[node_id]: inv
                for node_id, inv in node_inventories.items()
            }
            for product, node_inventories in inventory.items()
        }

        # Precompute distances
        distances = np.array([
            [fo.compute_distance(
                node.zip_code,
                order.destination_zip, zip_lat_longs)
             for node in nodes]
            for order in tqdm(orders)
        ])
        node_index_near_to_far = np.argsort(distances, axis=1)
        distances = np.sort(distances, axis=1)
        num_nodes = len(nodes)

        for date in capacity.keys():
            capacity[date] = np.array(
                [capacity[date].get(node.node_id, 0)
                 for node in nodes])
        num_products = len(inventory.keys())

        init_inventory = np.zeros((num_products, num_nodes))
        for i, product in enumerate(inventory.keys()):
            for j, node in enumerate(nodes):
                init_inventory[i,j] = inventory[product].get(
                    node.node_id, 0)
        return cls(
            inventory, capacity, orders, nodes,
            distances, node_index_near_to_far, init_inventory,
            num_products
        )    

@dataclass
class Update:
    indices: NDArray[np.int_]  # Timesteps to update
    values : NDArray[np.int_]  # New update values


@dataclass
class NewProblem:
    inventory             : np.ndarray
    capacity              : np.ndarray
    order_dates           : np.ndarray
    order_products        : np.ndarray
    distances             : np.ndarray  
    node_index_near_to_far: np.ndarray
    num_products          : int
    num_nodes             : int

    def __init__(self, inventory, capacity, order_dates, order_products, distances, node_index_near_to_far, num_products, num_nodes, order_locations = None):
        self.inventory = inventory
        self.capacity = capacity
        self.order_dates = order_dates
        self.order_products = order_products
        self.distances = distances
        self.node_index_near_to_far = node_index_near_to_far
        self.num_products = num_products
        self.num_nodes = num_nodes
        self.order_locations = order_locations

    @classmethod
    def load(cls, directory, canonical_num_products, sample_path_index=0):
        data = fo.initialize(directory, sample_path_index)
        inventory = data["inventory"]
        capacity = data["capacity"]
        orders = data["orders"]
        nodes = data["nodes"]
        zip_lat_longs = data["zip_lat_longs"]

        def extract_product_id(product_name):
            return int(product_name.split('_')[1])

        node_id_to_int = {
            node['node_id']: i for i, node in enumerate(nodes)
        }

        num_products = canonical_num_products
        num_nodes = len(nodes)

        # Convert dicts into objects, and convert str ids to ints
        init_date = orders[0]['timestamp'].date().toordinal()
        order_dates = [order['timestamp'].date().toordinal() - init_date for order in orders]
        order_products = [extract_product_id(order['product']) for order in orders]

        inventory_out = np.zeros((num_products, num_nodes))
        for product, node_inventories in inventory.items():
            for node_id, inv in node_inventories.items():
                inventory_out[extract_product_id(product), node_id_to_int[node_id]] = inv

        capacity_out = np.zeros((order_dates[-1]+1, num_nodes))
        for date, node_capacities in capacity.items():
            for node_id, capacity in node_capacities.items():
                capacity_out[date.toordinal() - init_date, node_id_to_int[node_id]] = capacity

        # Precompute distances

        np.random.seed(0)
        distances = np.array([
            [fo.compute_distance(
                node['zip_code'],
                order['destination_zip'], zip_lat_longs) + np.random.exponential(0.1) #add noise to the distance to (hopefully) guarantee the unique optimal solution for LP
             for node in nodes]
            for order in orders])
        node_index_near_to_far = np.argsort(distances, axis=1)
        # distances = np.sort(distances, axis=1)
        return cls(
            inventory_out, capacity_out,
            np.array(order_dates), np.array(order_products),
            distances,
            node_index_near_to_far,
            num_products, num_nodes,
            order_locations = [order['destination_zip'] for order in orders]
        )

@dataclass
class NewUpdate:
    t        : int  # Timestep to update from
    product  : int  # Product to update
    update_fulfill_index  :  np.ndarray
    update_fulfill_value  :  np.ndarray
