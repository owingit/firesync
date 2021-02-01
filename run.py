import json
import math
import networkx as nx
import os
import sys
import csv
import numpy as np
from datetime import datetime

import multiprocessing
import pathos.multiprocessing as multip
import argparse

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
E_DELTAS = "epsilon_deltas"
BETAS = "betas"
DBS = "databases"
PHRASE_DURATIONS = "phrases"

TRACE_KEY = 'all_paths'
FLASH_KEY = 'all_flash_steps'
OBSTACLE_KEY = 'obstacles'
DISTANCE_KEY = 'distances'

USE_KURAMATO = False
DUMP_DATA = True
DO_PLOTTING = True


def main():
    now = datetime.now()
    experiment_results = {}

    # can pass 1 or more db files without specifying any other arguments
    if len(sys.argv) > 1 and "-n" not in sys.argv:
        process_json_db(sys.argv, experiment_results)

    # can also pass multiple arguments to run new simulations (agent count, side len, simulation len, trials)
    elif len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--num", "-n", type=int, nargs='+', required=True)
        parser.add_argument("--steps", "-s", type=int, required=True)
        parser.add_argument("--length", "-l", type=int, required=True)
        parser.add_argument("--trials", "-t", type=int, required=True)
        parser.add_argument("--beta_range", "-b", type=float, nargs='+', required=False)
        parser.add_argument("--epsilon_delta_range", "-e", type=float, nargs='+', required=False)
        parser.add_argument('--obstacles', dest='obstacles', action='store_true')
        parser.set_defaults(obstacles=False)
        args = parser.parse_args()
        beta_range = extract_range(args.beta_range)
        epsilon_deltas = extract_range(args.epsilon_delta_range)
        use_obstacles = args.obstacles
        num_list = [int(num) for num in args.num]
        params = set_constants(nao=num_list, sc=args.steps, sl=args.length, nt=args.trials, betas=beta_range,
                               epsilon_deltas=epsilon_deltas)

        simulations = setup_simulations(params, use_obstacles=use_obstacles)
        experiment_results = run_simulations(simulations, use_processes=True)
        if DUMP_DATA:
            write_results(experiment_results, now)

    # or run the default settings
    else:
        params = set_constants()
        simulations = setup_simulations(params)
        experiment_results = run_simulations(simulations, use_processes=True)
        if DUMP_DATA:
            write_results(experiment_results, now)
    if DO_PLOTTING:
        plotter = sp.Plotter(experiment_results, now)
        # plotter.plot_example_animations()
        # plotter.compare_obstacles_vs_no_obstacles()
        plotter.plot_quiet_period_distributions(on_betas=False)
        if USE_KURAMATO:
            plotter.plot_mean_vector_length_results()
    print("done")


def extract_range(arglist):
    """Process argument list. Arglist can either be empty, one value, or two values constituting a min-max range."""
    if arglist is not None:
        float_args = [float(a) for a in arglist]
        if len(float_args) == 1:
            retlist = float_args
        else:
            retlist = np.arange(float_args[0], float_args[1], 0.01)
    else:
        retlist = None
    return retlist


def process_json_db(program_argv, experiment_results):
    """Process data from folder or database file

    :param program_argv: argv
    :param experiment_results: dict to update
    """
    for db in program_argv[1:]:
        if os.path.isdir(os.path.abspath(db)):
            for dbf in os.listdir(os.path.abspath(db)):
                obstacle_flag = False
                if 'obstacles' in dbf:
                    obstacle_flag = True
                raw_experiment_results = load_experiment_results(os.path.abspath(db + '/' + dbf))
                if raw_experiment_results:
                    experiment_results.update(
                        process_results_from_written_file(raw_experiment_results, obstacle_flag))
                else:
                    raise TypeError("json data expected!")
        else:
            obstacle_flag = False
            if 'obstacles' in db:
                obstacle_flag = True
            raw_experiment_results = load_experiment_results(os.path.abspath(db))
            if raw_experiment_results:
                experiment_results.update(process_results_from_written_file(raw_experiment_results, obstacle_flag))
            else:
                raise TypeError("json data expected!")


def load_experiment_results(db_file):
    if '.json' in db_file:
        with open(db_file, 'rb+') as data:
            json_dict = json.load(data)
        return json_dict
    else:
        print('Experiment results must be in .json format!')
        return None


def process_results_from_written_file(raw_experiment_results, if_obstacles, if_kuramato=USE_KURAMATO):
    name = list(raw_experiment_results.keys())[0]
    retdict = {name: []}

    for index in range(len(raw_experiment_results[name])):
        dummy_simulation = create_dummy_simulation_from_raw_experiment_results(
            name, index, raw_experiment_results, if_obstacles, if_kuramato
        )
        retdict[name].append(dummy_simulation)
    return retdict


def create_dummy_simulation_from_raw_experiment_results(name, index, raw_experiment_results, if_obstacles, if_kuramato):
    num_steps = len(raw_experiment_results[name][0].get(TRACE_KEY)[0].keys())
    data = raw_experiment_results[name][index]
    num_agents = len(list(data.get(TRACE_KEY)))
    beta = float(name.split('beta')[0].split('density')[1])
    density = float(name.split('beta')[0].split('density')[0])
    side_length = math.sqrt((float(num_agents)) / density)
    phrase_duration = name.split('beta')[1].split('Tb')[0]
    epsilon_delta = 0.3333 # placeholder
    if phrase_duration != 'distribution':
        phrase_duration = int(phrase_duration)

    dummy_simulation = Simulation.Simulation(num_agents=num_agents,
                                             side_length=side_length, step_count=num_steps, thetastar=math.pi * 2,
                                             coupling_strength=0.03,
                                             Tb=1.57,
                                             beta=beta, phrase_duration=phrase_duration, r_or_u="uniform",
                                             epsilon_delta=epsilon_delta,
                                             use_obstacles=if_obstacles, use_kuramato=if_kuramato)
    dummy_simulation.has_run = True

    for i, firefly in enumerate(dummy_simulation.firefly_array):
        firefly.trace = data.get(TRACE_KEY)[i]
        for step, p in firefly.trace.items():
            firefly.positionx[int(step)] = p[0]
            firefly.positiony[int(step)] = p[1]
        firefly.flashed_at_this_step = data.get(FLASH_KEY)[i]

    if type(data.get(OBSTACLE_KEY)) == list:
        dummy_simulation.obstacles = [ob.Obstacle(blob[0], blob[1], blob[2])
                                      for blob in data.get(OBSTACLE_KEY)]
    return dummy_simulation


def set_constants(sl=None, sc=None, nao=None, nt=None, betas=None, epsilon_deltas=None):
    if not sl:
        side_length = 16
    else:
        side_length = sl
    if not nao:
        num_agent_options = [20]
    else:
        num_agent_options = nao
    if not sc:
        step_count = 4000
    else:
        step_count = sc
    if not nt:
        num_trials = 5
    else:
        num_trials = nt
    if betas is None:
        btas = [0.2]
    else:
        btas = betas
    if epsilon_deltas is None:
        epdeltas = [0.333333]
    else:
        epdeltas = epsilon_deltas
    params = {}
    thetastars = [2 * math.pi]
    inter_burst_intervals = [1.57]  # radians / sec
    coupling_strengths = [0.03]  # , 0.2, 0.5]
    params[PHRASE_DURATIONS] = ["distribution"]
    params[BETAS] = btas
    params[TSTARS] = thetastars
    params[TBS] = inter_burst_intervals
    params[NS] = side_length
    params[NUM_AGENTS] = num_agent_options
    params[STEPS] = step_count
    params[KS] = coupling_strengths
    params[TRIALS] = num_trials
    params[E_DELTAS] = epdeltas
    return params


def write_results(experiment_results, now):
    write_networks = False
    for k in experiment_results.values():
        dict_to_dump = {}
        name = ''
        for experiment in k:
            result = [x.strip() for x in experiment.boilerplate.split(',')]
            name = result[0] + result[1] + result[2] + '{}_steps'.format(experiment.steps)
            if experiment.obstacles:
                obs = [(obstacle.centerx, obstacle.centery, obstacle.radius) for obstacle in experiment.obstacles]
            else:
                obs = 'No obstacles'
            if dict_to_dump.get(name):
                dict_to_dump[name].append({
                    TRACE_KEY: [ff_i.trace for ff_i in experiment.firefly_array],
                    FLASH_KEY: [ff.flashed_at_this_step for ff in experiment.firefly_array],
                    DISTANCE_KEY: experiment.distance_statistics,
                    OBSTACLE_KEY: obs
                })
            else:
                dict_to_dump[name] = [{
                    TRACE_KEY: [ff_i.trace for ff_i in experiment.firefly_array],
                    FLASH_KEY: [ff.flashed_at_this_step for ff in experiment.firefly_array],
                    DISTANCE_KEY: experiment.distance_statistics,
                    OBSTACLE_KEY: obs
                }]
            if write_networks:
                write_network_data(experiment, now)
        with open("data/raw_experiment_results/{}_experiment_results_{}.json".format(
                name, str(now).replace(' ', '_')), 'w') as f:
            json.dump(dict_to_dump, f)


def write_network_data(experiment, now):
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


def setup_simulations(params, use_obstacles=False):
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
                    for epsilon_delta in params[E_DELTAS]:
                        for beta in params[BETAS]:
                            for phrase_duration in params[PHRASE_DURATIONS]:
                                for trial in range(0, params[TRIALS]):
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
                                                                       epsilon_delta=epsilon_delta,
                                                                       r_or_u="random",
                                                                       use_obstacles=use_obstacles,
                                                                       use_kuramato=USE_KURAMATO)
                                    simulations.append(simulation)
    return simulations


def run_simulations(simulations, use_processes=True):
    """
    Run all simulations set up by setup_simulations.
    The results are stored in a dictionary keyed by their parameters.
    """
    experiment_results = {}
    if use_processes:
        process_pool = multip.ProcessingPool(multip.cpu_count())
        process_results = process_pool.map(run_simulation_in_process, simulations)

        for finished_simulation in process_results:
            if finished_simulation.use_kuramato:
                result_key = frozenset((finished_simulation.tstar_seed,
                                        finished_simulation.Tb,
                                        finished_simulation.total_agents,
                                        finished_simulation.coupling_strength))
            else:
                result_key = frozenset((finished_simulation.beta,
                                        finished_simulation.phrase_duration,
                                        finished_simulation.total_agents,
                                        finished_simulation.n))
            if experiment_results.get(result_key):
                experiment_results[result_key].append(finished_simulation)
            else:
                experiment_results[result_key] = [finished_simulation]

    else:
        for simulation in simulations:
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


def run_simulation_in_process(simulation):
    print('running: with {} agents'.format(simulation.total_agents))
    simulation.run()
    return simulation


if __name__ == "__main__":
    main()
