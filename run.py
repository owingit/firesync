import json
import math
import sys
from datetime import datetime

import matplotlib.pyplot as plt

import Simulation
import obstacle as ob

TSTARS = "theta*"
TBS = "Tb"
NS = "n"
NUM_AGENTS = "num_agents"
STEPS = "steps"
KS = "k"
TRIALS = "trials"
BETAS = "betas"
PHRASE_DURATIONS = "phrases"

PHASE_KEY = 'all_phases'
VOLTAGE_KEY = 'all_voltages'
TRACE_KEY = 'all_paths'
FLASH_KEY = 'all_flash_steps'
OBSTACLE_KEY = 'obstacles'
DISTANCE_KEY = 'distances'

USE_KURAMATO = False


def main():
    params = set_constants()
    now = datetime.now()
    experiment_results = {}
    if len(sys.argv) > 1:
        for db in sys.argv[1:]:
            obstacle_flag = False
            if 'obstacles' in db:
                obstacle_flag = True
            raw_experiment_results = load_experiment_results(db)
            experiment_results.update(process_results_from_file(raw_experiment_results, params, obstacle_flag))
    else:
        simulations = setup_simulations(params)
        experiment_results = run_simulations(simulations)
        write_results(experiment_results, now)

    plot_animations(experiment_results, now)
    compare_obstacles_vs_no_obstacles(experiment_results, now)
    print("done")
    if USE_KURAMATO:
        plot_mean_vector_length_results(params, experiment_results)


def write_results(experiment_results, now):
    for k in experiment_results.values():
        for experiment in k:
            result = [x.strip() for x in experiment.boilerplate.split(',')]
            name = result[0] + result[1] + result[2] + '{}_steps'.format(experiment.steps)
            dict_to_dump = {
                name:
                {
                    TRACE_KEY: [ff_i.trace for ff_i in experiment.firefly_array],
                    FLASH_KEY: [ff.flashed_at_this_step for ff in experiment.firefly_array],
                    PHASE_KEY: [firefly.phase.tolist() for firefly in experiment.firefly_array],
                    VOLTAGE_KEY: [f.voltage_instantaneous.tolist() for f in experiment.firefly_array],
                    DISTANCE_KEY: experiment.distance_statistics
                }
            }
            if experiment.obstacles:
                dict_to_dump[name][OBSTACLE_KEY] = [(obstacle.centerx, obstacle.centery, obstacle.radius)
                                                    for obstacle in experiment.obstacles]
            with open("data/raw_experiment_results/{}_experiment_results_{}.json".format(name, now), 'w') as f:
                json.dump(dict_to_dump, f)


def load_experiment_results(db_file):
    with open(db_file, 'rb+') as data:
        json_dict = json.load(data)
        print(json_dict)
    return json_dict


def process_results_from_file(raw_experiment_results, params, if_obstacles, if_kuramato=USE_KURAMATO):
    print(raw_experiment_results)
    name = list(raw_experiment_results.keys())[0]
    num_agents = len(list(raw_experiment_results[name].get(TRACE_KEY)))
    num_steps = len(list(raw_experiment_results[name].get(TRACE_KEY)[0].keys()))
    beta = float(name.split('beta')[0].split('density')[1])
    phrase_duration = int(name.split('beta')[1].split('Tb')[0])
    dummy_simulation = Simulation.Simulation(num_agents=num_agents,
                                             side_length=params[NS], step_count=num_steps, thetastar=math.pi * 2,
                                             coupling_strength=0.03,
                                             Tb=1.57,
                                             beta=beta, phrase_duration=phrase_duration, r_or_u="uniform",
                                             use_obstacles=if_obstacles, use_kuramato=if_kuramato)
    dummy_simulation.has_run = True

    for i, firefly in enumerate(dummy_simulation.firefly_array):
        firefly.phase = raw_experiment_results[name].get(PHASE_KEY)[i]
        firefly.trace = raw_experiment_results[name].get(TRACE_KEY)[i]
        for step, p in firefly.trace.items():
            firefly.positionx[int(step)] = p[0]
            firefly.positiony[int(step)] = p[1]
        firefly.flashed_at_this_step = raw_experiment_results[name].get(FLASH_KEY)[i]
        firefly.voltage_instantaneous = raw_experiment_results[name].get(VOLTAGE_KEY)[i]

    if raw_experiment_results[name].get(OBSTACLE_KEY):
        dummy_simulation.obstacles = [ob.Obstacle(blob[0], blob[1], blob[2])
                                      for blob in raw_experiment_results[name].get(OBSTACLE_KEY)]

    if if_kuramato:
        result_key = frozenset((dummy_simulation.tstar_seed,
                                dummy_simulation.Tb,
                                dummy_simulation.total_agents,
                                dummy_simulation.coupling_strength,
                                if_obstacles))
    else:
        result_key = frozenset((dummy_simulation.beta,
                                dummy_simulation.phrase_duration,
                                dummy_simulation.total_agents,
                                dummy_simulation.n,
                                if_obstacles))
    return {result_key: [dummy_simulation]}


def set_constants():
    params = {}
    thetastars = [2 * math.pi]
    inter_burst_intervals = [1.57]  # radians / sec
    side_length = 25
    num_agent_options = [36]  # , 500, 1000]
    step_count = 1600
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
                                                                   beta=beta,
                                                                   phrase_duration=phrase_duration,
                                                                   r_or_u="random",
                                                                   use_obstacles=use_obstacles,
                                                                   use_kuramato=USE_KURAMATO)
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
        if simulation.use_kuramato:
            result_key = frozenset((simulation.tstar_seed,
                                    simulation.Tb,
                                    simulation.total_agents,
                                    simulation.coupling_strength))
        else:
            result_key = frozenset((simulation.beta,
                                    simulation.phrase_duration,
                                    simulation.total_agents,
                                    simulation.n))
        if experiment_results.get(result_key):
            experiment_results[result_key].append(simulation)
        else:
            experiment_results[result_key] = [simulation]
    return experiment_results


def plot_animations(experiment_results, now):
    """Call a simulation's animation functionality."""
    for identifier, simulation_list in experiment_results.items():
        for simulation in simulation_list:
            if simulation.use_kuramato:
                simulation.animate_phase_bins(now, show_gif=False, write_gif=True)
        for simulation in simulation_list:
            simulation.animate_walk(now, show_gif=True, write_gif=False)
            simulation.plot_bursts(now, show_gif=True, write_gif=False)


def compare_obstacles_vs_no_obstacles(experiment_results, now):
    """Plot obstacle simulations against no obstacles."""
    obstacle_simulations = []
    non_obstacle_simulations = []
    value = list(experiment_results.values())[0][0]
    num_agents = value.total_agents
    steps = value.steps
    bursts_axis = plt.axes(xlim=(0, steps), ylim=(0, num_agents))
    show = True
    write = False
    for identifier, simulation_list in experiment_results.items():
        for simulation in simulation_list:
            if simulation.obstacles:
                obstacle_simulations.append(simulation)
            else:
                non_obstacle_simulations.append(simulation)
            simulation.plot_bursts(now, show_gif=show, write_gif=write, shared_ax=bursts_axis)
    assert len(obstacle_simulations) == len(non_obstacle_simulations), 'Need both obstacle and non obstacle sims!'
    plt.legend()
    plt.title('Flashes over time ' + value.boilerplate + ' with and without obstacles')
    if show:
        plt.show()
    if write:
        save_string = value.set_save_string('flashplot', now)
        save_string = 'combined' + save_string
        plt.savefig(save_string)


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
