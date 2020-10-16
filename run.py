import numpy as np
import sys
import Simulation
import math
import matplotlib.pyplot as plt
import json
from datetime import datetime


TSTARS = "theta*"
TBS = "Tb"
NS = "n"
NUM_AGENTS = "num_agents"
STEPS = "steps"
KS = "k"
TRIALS = "trials"
BETAS = "betas"
PHRASE_DURATIONS = "phrases"


def main():
    params = set_constants()
    if len(sys.argv) > 1:
        db = sys.argv[1]
        experiment_results = load_experiment_results(db)
    else:
        simulations = setup_simulations(params)
        experiment_results = run_simulations(simulations)
    plot_animations(experiment_results)
    for k in experiment_results.values():
        for experiment in k:
            result = [x.strip() for x in experiment.boilerplate.split(',')]
            name = result[0] + result[1] + result[2]
            dict_to_dump = {name: {
                'all_paths': [ff_i.trace for ff_i in experiment.firefly_array],
                'all_flash_steps': [ff.flashed_at_this_step for ff in experiment.firefly_array],
                'all_phases': [firefly.phase.tolist() for firefly in experiment.firefly_array],
                'all_voltages': [f.voltage_instantaneous.tolist() for f in experiment.firefly_array]
            }}
            json.dump(dict_to_dump,
                      open("data/raw_experiment_results/{}_experiment_results_{}.json".format(name,datetime.now()),
                           'w'))

    plot_mean_vector_length_results(params, experiment_results)


def load_experiment_results(db_file):
    with open(db_file, 'w+') as data:
        json_dict = json.load(data)
        print(json_dict)
    return json_dict


def set_constants():
    params = {}
    thetastars = [2 * math.pi]
    inter_burst_intervals = [1.57]  # radians / sec
    side_length = 25
    num_agent_options = [25]  # , 500, 1000]
    step_count = 1000
    coupling_strengths = [0.03]  # , 0.2, 0.5]
    num_trials = 2
    params[PHRASE_DURATIONS] = [190]
    params[BETAS] = [0.1]
    params[TSTARS] = thetastars
    params[TBS] = inter_burst_intervals
    params[NS] = side_length
    params[NUM_AGENTS] = num_agent_options
    params[STEPS] = step_count
    params[KS] = coupling_strengths
    params[TRIALS] = num_trials
    return params


def setup_simulations(params):
    """
    Instantiate t*n*cs*tb*trial simulation objects with their parameters, where

    t=number of different thetastar ranges,
    n=number of different agent counts,
    cs=number of different coupling strengths,
    tb=number of different internal frequencies,
    trial=number of trials. All these values are held in the params dict.
    Right now, side length and step count are held as constants, but the params dict could easily pass those as lists
    and add to the combinatorics by iterating through each of those as well.

    """
    simulations = []
    for thetastar in params[TSTARS]:
        for num_agents in params[NUM_AGENTS]:
            for coupling_strength in params[KS]:
                for Tb in params[TBS]:
                    for beta in params[BETAS]:
                        for phrase_duration in params[PHRASE_DURATIONS]:
                            for trial in range(0, params[TRIALS]):
                                if trial % 2 == 0:
                                    use_obstacles = True
                                else:
                                    use_obstacles = False
                                n = params[NS]
                                step_count = params[STEPS]
                                simulation = Simulation.Simulation(num_agents=num_agents,
                                                                   side_length=n,
                                                                   step_count=step_count,
                                                                   thetastar=thetastar,
                                                                   coupling_strength=coupling_strength,
                                                                   Tb=Tb,
                                                                   beta=beta,
                                                                   phrase_duration=phrase_duration,
                                                                   r_or_u="random",
                                                                   use_obstacles=use_obstacles,
                                                                   use_kuramato=False)
                                simulations.append(simulation)
    return simulations


def run_simulations(simulations):
    """
    Run all simulations set up by setup_simulations.
    The results are stored in a dictionary keyed by their parameters.
    """
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
    """Call a simulation's animation functionality."""
    now = datetime.now()
    for identifier, simulation_list in experiment_results.items():
        for simulation in simulation_list:
            if simulation.use_kuramato:
                simulation.animate_phase_bins(now, show_gif=False, write_gif=True)
                simulation.animate_walk(now, show_gif=False, write_gif=True)
            else:
                simulation.plot_bursts(now, show_gif=False, write_gif=True)
                simulation.animate_walk(now, show_gif=False, write_gif=True)


def plot_mean_vector_length_results(params, experiment_results):
    """Directly plot statistical results from a simulation."""
    ax = plt.axes(xlim=(0, params[STEPS] + 1), ylim=(0, 1.05))
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean resultant vector length')
    for identifier, simulations in experiment_results.items():
        _to_plot_obstacles = {key: 0 for key in simulations[0].mean_resultant_vector_length.keys()}
        _to_plot_no_obstacles = {key: 0 for key in simulations[0].mean_resultant_vector_length.keys()}
        for instance in simulations:
            if instance.use_obstacles:
                label = 'obstacles'
            else:
                label = 'no_obstacles'
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


if __name__ == "__main__":
    main()
