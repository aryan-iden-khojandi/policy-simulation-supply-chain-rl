import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from scipy.special import softmax
from datamodel import NewProblem
from copy import deepcopy
import pulp
from scipy.optimize import minimize
import cvxpy as cp

class Order:
    def __init__(self, N, J):
        self.product_id = np.random.randint(0, N)  # The product this order is for
        self.node_id = -1  # Node this order is assigned to, -1 indicates unassigned
        self.reward = 0.0
        self.distance = np.random.rand(J+1)
        self.distance[-1] = 100000  # Distance to unfulfilled order


def generate_orders(N, J, num_orders=10):
    orders = []
    for _ in range(num_orders):
        order = Order(N, J)
        orders.append(order)
    return orders


def compute_transformed_distance_score(all_distances, distance_for_node):

    return (all_distances.max() - distance_for_node) / all_distances.max() # all_distance.max() is the distance to the unfulfilled node


def compute_reward_for_order(order, capacity):
    # Calculate the score for each node
    possible_nodes = range(len(capacity))  # In future, this can depend on other factors such as shipping eligibility

    ## reward is the normalized (1 - distance), normalized by unfulfilled distance
    # rewards = (order.distance[-1] - order.distance[possible_nodes]) / order.distance[-1]
    rewards = compute_transformed_distance_score(order.distance, order.distance[possible_nodes])

    return rewards, possible_nodes


def assign_node_to_order_with_regularization(order, inventory, capacity, inventory_dual_variables,
                                             capacity_dual_variables, enforce_inv_cap_constraints, deterministic=True):

    product = order.product_id
    rewards, possible_nodes = compute_reward_for_order(order, capacity)
    adjusted_rewards = rewards - inventory_dual_variables[product, possible_nodes] - \
        capacity_dual_variables[possible_nodes]

    if enforce_inv_cap_constraints:
        valid_nodes = (inventory[order.product_id] > 0) & (capacity > 0)
        valid_nodes = valid_nodes[:-1]
    else:
        valid_nodes = np.array([True] * (len(possible_nodes)-1))

    if not any(valid_nodes):
        optimal_node = len(possible_nodes) - 1
        optimal_reward = rewards[optimal_node]  # Use the actual reward for updating dual variables

        order.node_id = optimal_node
        order.reward = optimal_reward

        return order, inventory_dual_variables, capacity_dual_variables

    rewards = rewards[:-1]
    possible_nodes = possible_nodes[:-1]
    adjusted_rewards = adjusted_rewards[:-1]

    epsilon = 1.e-5

    # Define the objective function to be minimized (-ve of the maximization problem)
    def objective(x):
        return -np.sum(adjusted_rewards * x) + epsilon * np.sum(x ** 2)

    # Initial guess
    x0 = np.zeros(len(rewards))

    # Define the constraints (sum(x) <= 1 and x >= 0 for all x)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Change to 'eq' to enforce the equality
            {'type': 'ineq', 'fun': lambda x: x})  # x >= 0

    # Bounds for each x.  Force to zero for infeasible nodes.
    def get_bound(node_idx, feasible_nodes):
        if feasible_nodes[node_idx] > 0:
            return 0, None
        else:
            return 0, 0

    bounds = [get_bound(node_idx, valid_nodes)
              for node_idx in range(len(valid_nodes))]
    # bounds = [(0, None) for _ in range(len(rewards))]

    # Solve the problem
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

    # ## penalize infeasible nodes by -1, note that the unfulfilled node has rewards 0, so it will be preferred than the infeasible nodes
    # adjusted_rewards_feasible = adjusted_rewards * valid_nodes + -1 * (1 - valid_nodes)

    # probabilities = softmax(adjusted_rewards_feasible)
    # optimal_node = np.random.choice(len(probabilities), p=probabilities)

    #         optimal_node = np.argmax(adjusted_rewards_feasible)

    # from IPython import embed
    # embed()
    feasible_x = res.x * valid_nodes + -1.0 * (1 - valid_nodes)
    if any(valid_nodes > 0):
        # print(f"Adjusted rewards: {adjusted_rewards}")
        # print(f"Valid nodes: {valid_nodes}")
        feasible_indices = np.where(valid_nodes > 0)[0]
        # print(f"Feasible indices: {feasible_indices}")
        probs_feasible = feasible_x[feasible_indices]
        # print(f"Feasible probabilities: {probs_feasible}")

        # # Should be unnecessary, given constraints in QP
        # probs_feasible = np.maximum(probs_feasible, 0.0)
        # if np.max(probs_feasible) < 1e-6:
        #     probs_feasible = np.ones(len(probs_feasible))
        # probs_feasible = probs_feasible / np.sum(probs_feasible)

        # print(f"Feasible probabilities: {probs_feasible}")
        if deterministic:
            optimal_node_in_feasible_indices = np.argmax(probs_feasible)
        else:
            try:
                optimal_node_in_feasible_indices = np.random.choice(len(probs_feasible),
                                                                    p=probs_feasible)
            except:
                from IPython import embed
                embed()
            # print(f"Probabilities of considered nodes: {probs_feasible}")
            # print(f"Chosen node: {optimal_node_in_feasible_indices}")
        # print(f"Optimal node in feasible_indices: {optimal_node_in_feasible_indices}")
        optimal_node = feasible_indices[optimal_node_in_feasible_indices]
        # print(f"Optimal node: {optimal_node}")
    else:
        optimal_node = len(possible_nodes) - 1

    optimal_reward = rewards[optimal_node]  # Use the actual reward for updating dual variables

    order.node_id = optimal_node
    order.reward = optimal_reward

    # Update inventory and capacity dual variables
    if optimal_node != len(capacity) - 1:  # If the order is fulfilled
        if inventory[product, optimal_node] <= 0 or capacity[optimal_node] <= 0:
            # print("Error: Inventory or capacity is zero when it should not be")
            pass
        inventory[product, optimal_node] -= 1
        capacity[optimal_node] -= 1

    return order, inventory_dual_variables, capacity_dual_variables


def assign_node_to_order(order, alpha1, alpha2, beta, inventory, capacity, inventory_dual_vars, capacity_dual_vars,
                         inventory_original, capacity_original, enforce_inv_cap_constraints, return_rewards=False):
    """
    Assigns an order to a node based on the specified policy, updating inventory and capacity.
    
    Parameters:
    - order: The order to assign.
    - alpha1, alpha2, beta: Policy parameters.
    - inventory, capacity: J-dimensional arrays representing the inventory and capacity of each node.
    - inventory_dual_vars: Nested (Dict) (or (List)) of length N (num_products) in outer level and J (num_nodes) in inner level
    - capacity_dual_vars: (Dict) (or (List)) of length J (num_nodes)
    
    Returns:
    - Updates inventory and capacity by reducing the selected node's values by one.
    """
    inventory_dual_vars = deepcopy(inventory_dual_vars)
    capacity_dual_vars = deepcopy(capacity_dual_vars)
    
    product = order.product_id

    rewards, possible_nodes = compute_reward_for_order(order, capacity)
    
    # Adjust (decrease) rewards by dual variables for resources used (NOTE: Assumes single unit of single product)
    adjusted_rewards = rewards - inventory_dual_vars[product][possible_nodes] - capacity_dual_vars[possible_nodes]
    adjusted_rewards = np.maximum(adjusted_rewards, 0.0)
    
    #print("Adjusted rewards: ", adjusted_rewards)
    if enforce_inv_cap_constraints:
        valid_nodes = (inventory[product] > 0) & (capacity > 0)
    else:
        valid_nodes = np.array(possible_nodes)
    # valid_nodes = np.array(len(possible_nodes),)

    ## penalize infeasible nodes by -1, note that the unfulfilled node has rewards 0, so it will be preferred than the infeasible nodes
    adjusted_rewards_feasible = adjusted_rewards * valid_nodes + -1 * (1- valid_nodes)

    if any(valid_nodes[:-1] > 0):
        # print(f"Adjusted rewards: {adjusted_rewards}")
        # print(f"Valid nodes: {valid_nodes}")
        feasible_indices = np.where(valid_nodes[:-1] > 0)[0]
        optimal_node_in_feasible_indices = np.argmax(adjusted_rewards_feasible[feasible_indices])
        optimal_node = feasible_indices[optimal_node_in_feasible_indices]
    else:
        optimal_node = len(possible_nodes) - 1
    
    optimal_reward = rewards[optimal_node]  # Use the actual reward for updating dual variables
    
    order.node_id = optimal_node
    order.reward = optimal_reward

    # constraint_violation = False
    # Update inventory and capacity dual variables
    if optimal_node != len(capacity) - 1:   # If the order is fulfilled
        if inventory[product, optimal_node] <= 0 or capacity[optimal_node] <= 0:
            # constraint_violation = True
            # print("Error: Inventory or capacity is zero when it should not be")
            pass
            print(adjusted_rewards, valid_nodes, optimal_node, adjusted_rewards_feasible)
            print("Error: Inventory or capacity is zero when it should not be")
            optimal_node = len(capacity) - 1  # Assign to unfulfilled node
        inventory[product, optimal_node] -= 1
        capacity[optimal_node] -= 1
        inventory_dual_vars[product, optimal_node] *= (1.0 + 1.0 / \
                                                       max(alpha1 * inventory_original[product, optimal_node] + alpha2,
                                                           0.01))  # avoid division by zero
        inventory_dual_vars[product, optimal_node] += \
            beta * optimal_reward / (max((alpha1 * inventory_original[product, optimal_node] + alpha2),
                                         0.01))  # avoid division by zero
        
        capacity_dual_vars[optimal_node] *= (1.0 + 1.0 / max(alpha1 * capacity_original[optimal_node] + alpha2, 0.01)) # avoid division by zero
        capacity_dual_vars[optimal_node] += beta * optimal_reward / max(alpha1 * capacity_original[optimal_node] + alpha2, 0.01) # avoid division by zero
    
    if return_rewards:
        return order, inventory_dual_vars, capacity_dual_vars, rewards, adjusted_rewards
    return order, inventory_dual_vars, capacity_dual_vars


def assign_orders_with_primal_dual_policy(orders, alpha1, alpha2, beta, inventory, capacity, inventory_dual_vars,
                                          capacity_dual_vars, inventory_original, capacity_original,
                                          enforce_inv_cap_constraints=True, deterministic=True, with_regularization=True):
    """
    Assigns orders to nodes based on the current policy parameters and updates the orders' node_id.
    This function modifies the orders list in place.
    """
    orders_new = []
    for order in orders:
        # order, inventory_dual_vars, capacity_dual_vars = assign_node_to_order(
        #     order, alpha1, alpha2, beta, inventory, capacity, inventory_dual_vars, capacity_dual_vars,
        #     inventory_original, capacity_original, enforce_inv_cap_constraints)
        if with_regularization:
            order, inventory_dual_vars, capacity_dual_vars = assign_node_to_order_with_regularization(
                order, inventory, capacity, inventory_dual_vars, capacity_dual_vars, enforce_inv_cap_constraints, deterministic=deterministic)
        else:
            order, inventory_dual_vars, capacity_dual_vars = assign_node_to_order(
                order, alpha1, alpha2, beta, inventory, capacity, inventory_dual_vars, capacity_dual_vars,
                inventory_original, capacity_original, enforce_inv_cap_constraints)

        orders_new.append(order)
    return orders_new


def calculate_cost(orders):
    """
    Calculate the total cost based on the distances to the chosen nodes for each order.
    Unfulfilled orders incur a specified penalty.
    
    Parameters:
    - orders: A list of Order objects.
    - penalty: The penalty cost for each unfulfilled order. Default is 1.
    
    Returns:
    - The total cost, considering the distances of fulfilled orders and penalties for unfulfilled ones.
    """
    total_cost = 0.0
    fulfillment = 0
    J = len(orders[0].distance) - 1  # Number of nodes
    for order in orders:
        if order.node_id < J:  # Check if the order is assigned to a valid node
            fulfillment += 1
        total_cost += order.distance[order.node_id]  # Add the distance to the total cost
    average_cost = total_cost / len(orders)
    fulfillment_rate = fulfillment / len(orders)

    return average_cost, fulfillment_rate


def calculate_cumulative_reward(orders, penalty=1):
    """
    Calculate the total cost based on the distances to the chosen nodes for each order.
    Unfulfilled orders incur a specified penalty.
    
    Parameters:
    - orders: A list of Order objects.
    - penalty: The penalty cost for each unfulfilled order. Default is 1.
    
    Returns:
    - The total cost, considering the distances of fulfilled orders and penalties for unfulfilled ones.
    """
    
    rewards = map(lambda x: x.reward, orders)
    
    return sum(rewards)


def generate_orders_from_prob(prob, unfulfill_distance=100000):
    orders = []

    for i in range(len(prob.order_products)):
        order = Order(1, 1)
        order.product_id = prob.order_products[i]
        order.node_id = -1
#         order.distance = prob.distances[i, :] / max_distance # Normalize distances
        order.distance = prob.distances[i, :]
        order.distance = np.append(order.distance, unfulfill_distance)  # Special-case this down the line instead
        if prob.order_locations is not None:
            order.order_location = prob.order_locations[i] 
        orders.append(order)
    
    return orders


def naive_policy(orders, inventory, capacity):
    """
    Assigns orders to the first node that has the product in stock and has capacity.
    """

    for order in orders:
        rewards, _ = compute_reward_for_order(order, capacity)
        ### sort node by order.distance and send the product to the first node that has the product in stock and has capacity 
        sorted_node_index = np.argsort(order.distance)
        for node_index in sorted_node_index:
            if inventory[order.product_id, node_index] > 0 and capacity[node_index] > 0:
                order.node_id = node_index
                order.reward = rewards[node_index]
                inventory[order.product_id, node_index] -= 1
                capacity[node_index] -= 1
                break
    return orders


class ConstraintBuilder:
    def __init__(self):
        self.constraintList = []
        self.str2constr = {}

    def addConstr(self, expr, str_):
        self.constraintList.append(expr)
        self.str2constr[str_] = len(self.constraintList)-1

    def get(self):
        return self.constraintList

    def getConstr(self, str_):
        return self.constraintList[self.str2constr[str_]]


def offline_LP_CVXPY(orders, inventory, capacity):

    T = len(orders)
    J = len(capacity)
    N = len(inventory)
    # epsilon_reg = 0.05
    constraints = ConstraintBuilder()

    # Assuming the problem setup is already defined as shown in the previous messages

    # Decision variables (binary)
    x = cp.Variable((T, J), nonneg=True)

    # Define the objective function components and constraints as per the last example

    transformed_distance_scores_array = np.zeros((T, J))
    # Assuming you have defined a function to compute these, let's fill the matrix
    for t in range(T):
        for j in range(J):
            transformed_distance_scores_array[t, j] = compute_transformed_distance_score(orders[t].distance,
                                                                                         orders[t].distance[j])

    transformed_distance_scores = cp.Parameter((T, J), value=transformed_distance_scores_array)

    # Objective Function with Quadratic Regularization
    # Objective Function: Maximizing the transformed distance score
    objective = cp.Maximize(cp.sum(cp.multiply(transformed_distance_scores, x)))
    # objective = cp.Maximize(cp.sum(cp.multiply(transformed_distance_scores, x)) - epsilon_reg * cp.quad_form(x.flatten(), np.eye(T*J)))

    # Each order must be fulfilled exactly once
    for order_index in range(T):
        order_constraint = cp.sum(x[order_index, :]) <= 1
        #     constraints.append(order_constraint)
        constraints.addConstr(order_constraint, f"order_{order_index}")

    # Inventory constraints
    inventory_constraints = []
    for n in range(N):
        for j in range(J):
            relevant_terms = [x[t, j] for t in range(T) if orders[t].product_id == n]
            if len(relevant_terms) > 0:
                inventory_constraint = cp.sum(relevant_terms) <= inventory[n][j]
                #             constraints.append(inventory_constraint)
                constraints.addConstr(inventory_constraint, f"inventory_product_{n}_node_{j}")
                inventory_constraints.append(inventory_constraint)

    # Capacity constraints
    capacity_constraints = []
    for j in range(J):
        capacity_constraint = cp.sum(x[:, j]) <= capacity[j]
        #     constraints.append(capacity_constraint)
        constraints.addConstr(capacity_constraint, f"capacity_node_{j}")
        capacity_constraints.append(capacity_constraint)

    # Define the problem with the objective and constraints
    problem = cp.Problem(objective, constraints.get())

    # Solve the problem
    problem.solve(ignore_dpp=True, verbose=True, solver=cp.ECOS)

    # Optionally, print out the optimal value of the objective function
    print("Optimal value with regularization:", problem.value)

    # Print the solution
    if problem.status == cp.OPTIMAL:
        print("Solution is optimal.")
        for t in range(T):
            for j in range(J):
                if x[t, j].value > 0:
                    print(f"x[{t}, {j}] = {x[t, j].value}")
    else:
        print("Solution is not optimal.")

    return problem, constraints


def offline_LP_scipy(orders, inventory, capacity, debug=True):
    # Define the objective function

    T = len(orders)
    J = len(capacity)
    N = len(inventory)

    def objective_function(x):
        # total_cost = 0
        # for t in range(T):
        #     for j in range(J):
        #         total_cost += compute_transformed_distance_score(orders[t].distance, orders[t].distance[j]) * x[
        #             t * J + j]
        costs = [compute_transformed_distance_score(orders[t].distance, orders[t].distance[j])
                 for t in range(T)
                 for j in range(J)]

        return np.sum(costs * x)  # Maximizing, so negate the objective function

    # Define the equality constraint for order fulfillment
    def order_fulfillment_constraint(x):
        constraints = []
        for t in range(T):
            constraint = np.sum(x[t * J: (t + 1) * J]) - 1  # Each order is fulfilled exactly once
            constraints.append(constraint)
        return constraints

    # Define the inequality constraints for inventory and capacity
    def inventory_capacity_constraints(x):
        constraints = []
        for n in range(N):
            for j in range(J):
                constraint = inventory[n][j] - np.sum(x[t * J + j]
                                                      for t in range(T) if orders[t].product_id == n)
                constraints.append(constraint)
        for j in range(J):
            constraint = capacity[j] - np.sum(x[t * J + j]
                                              for t in range(T))
            constraints.append(constraint)
        return constraints

    # Initial guess
    initial_guess = np.zeros(T * J)

    # Bounds for variables (binary variables)
    bounds = [(0, 1)] * (T * J)

    # Define the optimization problem
    constraints = [
        {'type': 'eq', 'fun': order_fulfillment_constraint},
        {'type': 'ineq', 'fun': inventory_capacity_constraints}
    ]

    # Solve the problem
    result = minimize(objective_function, initial_guess, bounds=bounds, constraints=constraints,
                      method='SLSQP', options={'disp': True})

    # Print the solution
    if result.success:
        print("Solution is optimal.")
        for i, val in enumerate(result.x):
            if val > 0:
                print(f"x[{i}] = {val}")
    else:
        print("Solution is not optimal. Status:", result.message)

    return result


def offline_LP(orders, inventory, capacity, debug=True):
    """
    Solves the order fulfillment problem using offline linear programming.
    """

    T = len(orders)
    J = len(capacity)
    N = len(inventory)

    # Create the problem variable to contain the problem data
    problem = pulp.LpProblem("Order_Fulfillment_Problem", pulp.LpMaximize)

    # Variables
    x = pulp.LpVariable.dicts("x", ((t, j) for t in range(T) for j in range(J)),lowBound=0, upBound=None, cat=pulp.LpContinuous)

    # Objective Function
    problem += pulp.lpSum(compute_transformed_distance_score(orders[t].distance, orders[t].distance[j]) * x[t, j]
                          for t in range(T) for j in range(J)), "Total_Fulfillment_Cost"

    # Constraints
    # Each order is fulfilled exactly once
    for t in range(T):
        problem += pulp.lpSum(x[t, j] for j in range(J)) <= 1, f"Order_{t}_Fulfillment"

    # Inventory constraints
    for n in range(N):
        for j in range(J):
            problem += pulp.lpSum(x[t, j] for t in range(T) if orders[t].product_id == n) <= inventory[n][j], f"Inventory_{n}_{j}"

    # Capacity constraints
    for j in range(J):
        problem += pulp.lpSum(x[t, j] for t in range(T)) <= capacity[j], f"Capacity_{j}"

    # Solve the problem
    if debug:
        problem.solve(pulp.PULP_CBC_CMD(mip=False, msg=True))
    else:
        solver = pulp.PULP_CBC_CMD(msg=True)
        solver.options.extend([
            '-ratioGap', '1e-13',  # Sets the relative gap tolerance for optimality to 1e-10
            '-maxIt', '10000'      # Sets the maximum number of iterations to 10,000
            '-allowableGap', '1e-13' # Sets the absolute gap tolerance for optimality to 1e-10,
            '-primalT', '1e-13', # Sets the primal tolerance to 1e-10
            '-dualT', '1e-13' # Sets the dual tolerance to 1e-10
        ])
        problem.solve(solver)

    # Print the solution
    # for v in problem.variables():
    #     if v.varValue > 0:
    #         print(v.name, "=", v.varValue)

    # Check if the solution is optimal
    if debug:
        if pulp.LpStatus[problem.status] == 'Optimal':
            print("Solution is optimal.")
        else:
            print("Solution is not optimal. Status:", pulp.LpStatus[problem.status])

    return problem


def obtain_primal_dual_theoretical_params(orders, inventory, capacity, inventory_dual_vars, capacity_dual_vars, enforce_inv_cap_constraints=True):

    fulfillment_and_reward_info = [
        assign_node_to_order(order, 1.0, 0.0, 0.0, inventory.copy(), capacity.copy(), inventory_dual_vars.copy(),
                             capacity_dual_vars.copy(), inventory.copy(), capacity.copy(), enforce_inv_cap_constraints, return_rewards=True)
        for order in orders
    ]
    rewards_by_order = [_[3]
                        for _ in fulfillment_and_reward_info]

    kappa_by_order = [max(rewards[:-1])/(min(rewards[:-1])+1e-6)
                      for rewards in rewards_by_order]
    kappa = np.median(kappa_by_order)

    # Theoretical values
    min_inventory = np.min([_ for _ in inventory.flatten() if _ > 0])
    alpha_1 = 1.0 / (1.0 + np.log(kappa))
    alpha_2 = 0.0
    beta = kappa / ((1 + 1.0/min_inventory)**(min_inventory/alpha_1)-1.0)

    return alpha_1, alpha_2, beta


def report_primal_dual_policy(orders, alpha_1, alpha_2, beta, inventory, capacity, inventory_dual_vars,
                              capacity_dual_vars, enforce_inv_cap_constraints=True, deterministic=True, with_regularization=True):
    orders = assign_orders_with_primal_dual_policy(orders, alpha_1, alpha_2, beta, inventory, capacity,
                                                   inventory_dual_vars, capacity_dual_vars,
                                                   inventory.copy(), capacity.copy(),
                                                   enforce_inv_cap_constraints=enforce_inv_cap_constraints,
                                                   deterministic=deterministic, with_regularization = with_regularization)
    reward = calculate_cumulative_reward(orders)
    average_distance, fufillment_rate = calculate_cost(orders)

    print("reward: ", reward)
    print("average distances",average_distance)
    print("fulfillment_rate", fufillment_rate)
