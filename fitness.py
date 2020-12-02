import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def fitness_function(filename, simulation_filename):
    df = pd.read_csv(filename, names=["x", "y", "z", "t"])

    max_t = 0
    for index, row in df.iterrows():
        if row["t"] > max_t:
            max_t = row["t"]
    num_flashed_at_timestep = {k: 0 for k in np.arange(0.0, max_t, 1)}
    for index, row in df.iterrows():
        if num_flashed_at_timestep.get(row["t"]) is not None:
            num_flashed_at_timestep[row["t"]] += 1

    sorted_flashes_at_each_t = {k: num_flashed_at_timestep[k] for k in sorted(num_flashed_at_timestep)}

    df_sim = pd.read_csv(simulation_filename, names=["x", "y", "label", "t"])

    max_t_sim = 0
    for index, row in df_sim.iterrows():
        if row["t"] > max_t_sim:
            max_t_sim = row["t"]
    sim_num_flashed_at_timestep = {k: 0 for k in np.arange(0.0, max_t_sim, 1)}
    for index, row in df_sim.iterrows():
        if sim_num_flashed_at_timestep.get(row["t"]) is not None:
            sim_num_flashed_at_timestep[row["t"]] += 1

    sim_sorted_flashes_at_each_t = {k: sim_num_flashed_at_timestep[k] for k in sorted(sim_num_flashed_at_timestep)}

    max_ = max(num_flashed_at_timestep.values())
    max_sim = max(sim_num_flashed_at_timestep.values())
    max_val = max([max_, max_sim]) + 5
    ax = plt.axes(xlim=(0, max_t), ylim=(0, max_val))
    ax.plot(list(sorted_flashes_at_each_t.keys()), list(sorted_flashes_at_each_t.values()),
            label='Wild', color='blue')
    ax.plot(list(sim_sorted_flashes_at_each_t.keys()), list(sim_sorted_flashes_at_each_t.values()),
            label='Simulated', color='green')

    ax.set_xlabel('Step')

    ax.set_ylabel('Number of flashers at timestep')
    plt.title('Flashes over time {} {}'.format(simulation_filename.split('/')[0],
                                               simulation_filename.split('.')[0].split('/')[1]))
    plt.legend()

    plt.show()


fitness_function('raw_data/xyzt_wild.csv', 'data/raw_experiment_results/0.390625density0.1betadistributionTb_obstacles1600_steps_experiment_results_2020-11-09_11:05:25.960570_csv_labeled.csv')
