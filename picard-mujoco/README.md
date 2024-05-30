Code for Mujoco experiments accompanying the paper "Speeding up Policy Simulation in Supply Chain RL."

To run:

1. `poetry lock && poetry install` to install dependencies
2. `cd scripts`
3. `snakemake -j1 rollout_times` to generate outputs
   
Output files are of the format `results/{env}/{seed}/timing.cs  v`

