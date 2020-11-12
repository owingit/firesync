import json
import math
import networkx as nx
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt

import simulation_plotter as sp
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

    plotter = sp.Plotter(experiment_results, now, params)
    plotter.plot_animations()
    # plotter.compare_obstacles_vs_no_obstacles()
    plotter.plot_quiet_period_distributions()
    print("done")
    if USE_KURAMATO:
        plotter.plot_mean_vector_length_results()


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
            with open("data/raw_experiment_results/{}_experiment_results_{}.json".format(name,
                                                                                         str(now).replace(' ', '_')),
                      'w') as f:
                json.dump(dict_to_dump, f)
            if len(experiment.firefly_array) > 1:
                if experiment.obstacles:
                    end_folder = '/with_obstacles'
                else:
                    end_folder = '/without_obstacles'
                pickle_folder = "pickled_networks_{}_steps_{}density{}beta{}Tb".format(
                    experiment.steps, (len(experiment.firefly_array) / experiment.n ** 2), experiment.beta,
                    experiment.phrase_duration
                )
                f = str(now).split('-')
                s = f[0] + '_' + f[1] + '_' + f[2].split(' ')[0] + '_' + f[2].split(' ')[1].split(':')[0] + \
                    f[2].split(' ')[1].split(':')[1] + f[2].split(' ')[1].split(':')[2].split('.')[0]
                landing_dir = '{}{}'.format(s, end_folder)
                for i, cascade in experiment.networks_in_cascade_.items():
                    for j, network in enumerate(cascade):
                        if not os.path.exists('data/raw_experiment_results/{}/{}'.format(pickle_folder, landing_dir)):
                            os.makedirs('data/raw_experiment_results/{}/{}'.format(pickle_folder, landing_dir))
                        nx.write_gpickle(network,
                                         'data/raw_experiment_results/{}/{}/cascade_{}_network_{}.gpickle'.format(
                                             pickle_folder, landing_dir, i, j
                                         ))
                for e, accumulated_network in experiment.connected_temporal_networks.items():
                    nx.write_gpickle(accumulated_network,
                                     'data/raw_experiment_results/{}/{}/accumulated_network_cascade_{}.gpickle'.format(
                                         pickle_folder, landing_dir, e
                                     ))


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
    phrase_duration = name.split('beta')[1].split('Tb')[0]
    if phrase_duration != 'distribution':
        phrase_duration = int(phrase_duration)
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
                                str(dummy_simulation.total_agents),
                                dummy_simulation.n,
                                if_obstacles))
    return {result_key: [dummy_simulation]}


def set_constants():
    params = {}
    thetastars = [2 * math.pi]
    inter_burst_intervals = [1.57]  # radians / sec
    side_length = 16
    num_agent_options = [1, 4, 9, 16, 25, 64, 100]
    step_count = 3200
    coupling_strengths = [0.03]  # , 0.2, 0.5]
    num_trials = 5
    params[PHRASE_DURATIONS] = ["distribution"]
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
                                # if trial % 2 == 0:
                                #     use_obstacles = False
                                # else:
                                #     use_obstacles = True
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
                                                                   use_obstacles=True,
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


if __name__ == "__main__":
    main()
