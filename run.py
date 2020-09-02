import numpy as np
import Simulation
import math
import matplotlib.pyplot as plt
import collections
from datetime import datetime


TSTARS = "theta*"
TBS = "Tb"
NS = "n"
NUM_AGENTS = "num_agents"
STEPS = "steps"
KS = "k"
TRIALS = "trials"


def main():
    params = set_constants()
    simulations = setup_simulations(params)
    experiment_results = run_simulations(simulations)
    plot_animations(experiment_results)

    plot_mean_vector_length_results(params, experiment_results)


def set_constants():
    params = {}
    thetastars = [2 * math.pi]
    inter_burst_intervals = [1.57]  # radians / sec
    side_length = 16
    num_agent_options = [160]  # , 500, 1000]
    step_count = 100
    coupling_strengths = [0.03]  # , 0.2, 0.5]
    num_trials = 1
    params[TSTARS] = thetastars
    params[TBS] = inter_burst_intervals
    params[NS] = side_length
    params[NUM_AGENTS] = num_agent_options
    params[STEPS] = step_count
    params[KS] = coupling_strengths
    params[TRIALS] = num_trials
    return params


def setup_simulations(params):
    simulations = []
    for thetastar in params[TSTARS]:
        for num_agents in params[NUM_AGENTS]:
            for coupling_strength in params[KS]:
                for Tb in params[TBS]:
                    for trial in range(0, params[TRIALS]):
                        if trial == 0:
                            use_obstacles = False
                        else:
                            use_obstacles = True
                        n = params[NS]
                        step_count = params[STEPS]
                        simulation = Simulation.Simulation(num_agents=num_agents,
                                                           side_length=n,
                                                           step_count=step_count,
                                                           thetastar=thetastar,
                                                           coupling_strength=coupling_strength,
                                                           Tb=Tb,
                                                           r_or_u="random",
                                                           use_obstacles=use_obstacles)
                        simulations.append(simulation)
    return simulations


def run_simulations(simulations):
    experiment_results = {}
    for count, simulation in enumerate(simulations):
        simulation.run()
        result_key = frozenset((simulation.tstar_seed,
                                simulation.Tb,
                                simulation.total_agents,
                                simulation.coupling_strength))
        if experiment_results.get(result_key):
            experiment_results[result_key].append(simulation)
        else:
            experiment_results[result_key] = [simulation]
    return experiment_results


def plot_animations(experiment_results):
    now = datetime.now()
    for identifier, simulation_list in experiment_results.items():
        for simulation in simulation_list:
            simulation.animate_phase_bins(now, show_gif=True, write_gif=False)
            simulation.animate_walk(now, show_gif=True, write_gif=False)


def plot_mean_vector_length_results(params, experiment_results):
    ax = plt.axes(xlim=(0, params[STEPS] + 1), ylim=(0, 1.05))
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean resultant vector length')
    for identifier, simulations in experiment_results.items():
        _to_plot_obstacles = {key: 0 for key in simulations[0].mean_resultant_vector_length.keys()}
        _to_plot_no_obstacles = {key: 0 for key in simulations[0].mean_resultant_vector_length.keys()}
        for i, instance in enumerate(simulations):
            if i % 2 == 0:
                label = 'no_obstacles'
            else:
                label = 'obstacles'
            for k, v in instance.mean_resultant_vector_length.items():
                if label == 'obstacles':
                    _to_plot_obstacles[k] += v
                else:
                    _to_plot_no_obstacles[k] += v
        divisor = int(len(simulations) / 2)
        if divisor == 0:
            divisor = 1
        _to_plot_no_obstacles = {k: v / divisor for k, v in _to_plot_no_obstacles.items()}
        _to_plot_obstacles = {k: v / divisor for k, v in _to_plot_obstacles.items()}
        ax.plot(_to_plot_no_obstacles.keys(), _to_plot_no_obstacles.values(), label='no_obstacles')
        ax.plot(_to_plot_obstacles.keys(), _to_plot_obstacles.values(), label='obstacles')
        ax.legend()
    plt.show()
    print(experiment_results)


if __name__ == "__main__":
    main()
