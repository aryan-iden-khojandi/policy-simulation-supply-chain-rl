# supply-chain-simulation
A repository to replicate experiments of Picard Iteration in SCO contexts.

To generate synthetic data:

python synthetic_data.py with problem.num_days=1 problem.num_orders=300000 problem.num_products=100000 problem.num_nodes=30 problem.fulfillable_prop=0.8 problem.demand.distribution=power problem.demand.alpha=1.0 problem.reservation_setup=False problem.num_sample_paths=1 _seed=42

To run a sample path:

python jax_sim.py with algo.max_workers=10000 algo.max_products_per_worker=100 algo.max_iters=100000 algo.num_steps=100 canonical_num_products=1000000 output=test.csv fulfill_output=fulfillment_results_picard input_dir=input_dicts/3000000_orders_1000000_products_30_nodes_0.8_fulfillable_demand_distribution_power_demand_alpha_1.0 _seed=42

To run E2E RL:

python jax_E2E_RL.py with algo.max_iters=100000 algo.num_steps=1000000 canonical_num_products=1000000 input_dir=input_dicts/3000000_orders_1000000_products_30_nodes_0.8_fulfillable_demand_distribution_loglinear_demand_beta_0.0 _seed=42 optimizer.debug_print_step=1
