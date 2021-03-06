import math
import pickle
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.interpolate import make_interp_spline

import simulation_helpers


class Plotter:
    def __init__(self, experiment_results, now):
        self.experiment_results = experiment_results
        self.now = now

    def plot_quiet_period_distributions(self, on_betas=False, path=None):
        """Grabs distributions from data and pickles for later plotting."""
        use_pickle_plotter = True
        pickle_flag = False
        if 'null' in path:
            is_null = True
        else:
            is_null = False
        interburst_interval_distribution = {}
        swarm_interburst_interval_distribution = {}
        ob_interburst_interval_distribution = {}
        ob_swarm_interburst_interval_distribution = {}
        for identifier, simulation_list in self.experiment_results.items():
            if '_obstacles' in identifier:
                ob_interburst_interval_distribution[identifier] = {}
                ob_swarm_interburst_interval_distribution[identifier] = {}
                for simulation in simulation_list:
                    if not on_betas:
                        k = simulation.total_agents
                    else:
                        k = simulation.beta
                    if not ob_interburst_interval_distribution[identifier].get(k):
                        ob_interburst_interval_distribution[identifier][k] = [simulation.calc_interburst_distribution()]
                    else:
                        ob_interburst_interval_distribution[identifier][k].append(
                            simulation.calc_interburst_distribution())

                    if not ob_swarm_interburst_interval_distribution[identifier].get(k):
                        ob_swarm_interburst_interval_distribution[identifier][k] = [simulation.swarm_interburst_dist(
                            is_null)
                        ]
                    else:
                        ob_swarm_interburst_interval_distribution[identifier][k].append(simulation.swarm_interburst_dist(
                            is_null)
                        )
            else:
                interburst_interval_distribution[identifier] = {}
                swarm_interburst_interval_distribution[identifier] = {}
                for simulation in simulation_list:
                    if not on_betas:
                        k = simulation.total_agents
                    else:
                        k = simulation.beta
                    if not interburst_interval_distribution[identifier].get(k):
                        interburst_interval_distribution[identifier][k] = [simulation.calc_interburst_distribution()]
                    else:
                        interburst_interval_distribution[identifier][k].append(simulation.calc_interburst_distribution())

                    if not swarm_interburst_interval_distribution[identifier].get(k):
                        swarm_interburst_interval_distribution[identifier][k] = [simulation.swarm_interburst_dist(
                            is_null)
                        ]
                    else:
                        swarm_interburst_interval_distribution[identifier][k].append(simulation.swarm_interburst_dist(
                            is_null)
                        )
        if pickle_flag:
            with open(path+'/beta_sweep_individual.pickle', 'wb') as f_i:
                print('pickling {}+beta_sweep_individual.pickle...'.format(path))
                pickle.dump(interburst_interval_distribution, f_i)
            with open(path+'/beta_sweep_swarm.pickle', 'wb') as f_s:
                print('pickling {}+beta_sweep_swarm.pickle...'.format(path))
                pickle.dump(swarm_interburst_interval_distribution, f_s)

        if not use_pickle_plotter:
            if len(ob_interburst_interval_distribution.items()) > 0:
                # s_means, s_stds, i_means, i_stds = self.calc_means_stds(ob_interburst_interval_distribution,
                #                                                         ob_swarm_interburst_interval_distribution,
                #                                                         on_betas=on_betas)
                self._plot_histograms(ob_interburst_interval_distribution,
                                      ob_swarm_interburst_interval_distribution,
                                      on_betas=on_betas)
                self._plot_all_histograms(ob_interburst_interval_distribution,
                                          ob_swarm_interburst_interval_distribution,
                                          obs=True, on_betas=on_betas)
            else:
                # s_means, s_stds, i_means, i_stds = self.calc_means_stds(interburst_interval_distribution,
                #                                                         swarm_interburst_interval_distribution,
                #                                                         on_betas=on_betas)
                self._plot_histograms(interburst_interval_distribution, swarm_interburst_interval_distribution,
                                      on_betas=on_betas)
                self._plot_all_histograms(interburst_interval_distribution, swarm_interburst_interval_distribution,
                                          on_betas=on_betas)

    @staticmethod
    def _plot_all_histograms(individual, group, obs=False, on_betas=False):
        if not on_betas:
            independent_var = 'ff'
        else:
            independent_var = 'beta_20ff'
        niceify = False
        dicts = [individual, group]
        bin_counts = [5, 10, 15, 20, 25, 30]
        for bin_count in bin_counts:
            for i, d in enumerate(dicts):
                fig, ax = plt.subplots()
                ax.set_xlabel('Interburst interval [s]')
                ax.set_ylabel('Freq count')
                colors = [cm.jet(x) for x in np.linspace(0.0, 1.0, len(d.keys())+1)]
                ax.set_xlim(0, 100)
                identifier_data = {}
                colorindex = 0

                for identifier, results in d.items():
                    for k, iid_list in results.items():
                        iids = [x for iid in iid_list for x in iid]
                        if not identifier_data.get(k):
                            identifier_data[k] = iids
                        else:
                            identifier_data[k].append(iids)

                sorted_dict = {k: identifier_data[k] for k in sorted(identifier_data)}
                for k, data in sorted_dict.items():

                    xs = []
                    for e in data:
                        if type(e) is not list:
                            xs.append(e)
                        else:
                            for element in e:
                                xs.append(element)
                    y, bin_edges = np.histogram(xs, bins=bin_count, density=True)
                    ys = [height for height in y]
                    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                    if niceify:
                        x_nice = np.linspace(min(bin_centers), max(bin_centers), 300)
                        _nice = make_interp_spline(bin_centers, ys)
                        y_nice = _nice(x_nice)
                        y_np = np.asarray(y_nice)
                        low_values_flags = y_np < 0.0  # Where values are low
                        y_np[low_values_flags] = 0.0
                        ax.plot(x_nice, y_nice, label='{}_{}_{}_pts'.format(k, independent_var, len(xs)),
                                color=colors[colorindex],)
                    else:
                        ax.plot(bin_centers, ys, label='{}_{}_{}_pts'.format(k, independent_var, len(xs)),
                                color=colors[colorindex])
                    colorindex += 1

                if i == 0:
                    string = '{}_bins_Individual_avg'.format(bin_count)
                    if obs:
                        string = 'obs' + string
                else:
                    string = '{}_bins_Swarm_avg'.format(bin_count)
                    if obs:
                        string = 'obs' + string
                plt.title('{}_interburst_histograms'.format(string))
                plt.legend()
                plt.savefig('histograms/{}_interburst_histograms_{}_smoothed.png'.format(string, str(niceify)))
                plt.clf()
                plt.close()

    @staticmethod
    def _plot_histograms(individual, group, on_betas=False):
        s_means, s_stds, i_means, i_stds = simulation_helpers.calc_means_stds(individual,
                                                                              group,
                                                                              on_betas=on_betas)
        if not on_betas:
            independent_var = 'ff'
        else:
            independent_var = 'beta_20ff'
        dicts = [group]
        for i, d in enumerate(dicts):
            for identifier, results in d.items():
                fig, ax = plt.subplots()
                ax.set_xlabel('Interburst interval [s]')
                ax.set_ylabel('Freq count')
                for k, iid_list in results.items():
                    _iids = [x / 10 for iid in iid_list for x in iid]
                    iids = [i for i in _iids if i > 3]
                    if k == 1:
                        ax.set_xlim(0, 500)
                    else:
                        ax.set_xlim(0, 120)
                    x = []
                    y = []
                    y_maybes = []
                    datafile = 'data/raw_experiment_results/3_22/silo/10ff/ibs10ff.csv'
                    with open(datafile, 'r') as csvfile:
                        plots = csv.reader(csvfile, delimiter=',')
                        for i, row in enumerate(plots):
                            x.append(i)
                            y_maybe = float(row[0])
                            y_maybes.append(y_maybe)
                            if y_maybe > 3:
                                y.append(y_maybe)

                    # ccdfx, ccdfy = ccdf(np.array(y))
                    ys, bin_edges = np.histogram(y, bins=np.arange(0, 1000, 3.0), density=True)
                    y_heights = [height for height in ys]

                    y_heights.append(0.0)
                    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                    ax.hist(y, bins=np.arange(0, 1000, 3.0), color='red', edgecolor='black', label='experimental',
                            density=True,
                            alpha=0.5)
                    ax.hist(iids, density=True, bins=np.arange(0, 1000, 3.0), color='cyan', edgecolor='black', alpha=0.5)
                    trials = len(iids)
                    # if i == 0:
                    #     if type(i_means[k]) is not str:
                    #         mean = math.floor(i_means[k])
                    #         std = math.floor(i_stds[k])
                    #     else:
                    #         mean = 'xx'
                    #         std = 'xx'
                    #     if trials > 1:
                    #         string = 'Individual_avg_over_' + str(trials) + '_{}mean_{}std'.format(mean, std)
                    #     else:
                    #         string = 'Individual_avg' + '_{}mean_{}std'.format(mean, std)
                    #     if '_obstacles' in identifier:
                    #         string = 'obs' + string
                    # else:
                    if type(s_means[k]) is not str:
                        mean = math.floor(s_means[k])
                        std = math.floor(s_stds[k])
                    else:
                        mean = 'xx'
                        std = 'xx'
                    if trials > 1:
                        string = 'Swarm_avg_over_' + str(trials) + '_{}mean_{}std'.format(mean, std)
                    else:
                        string = 'Swarm_avg' + '_{}mean_{}std'.format(mean, std)
                    if '_obstacles' in identifier:
                        string = 'obs' + string
                    plt.title('{}{}_{}mean_{}std'.format(
                        k,
                        independent_var,
                        mean,
                        std
                        ))
                    plt.savefig('histograms/{}_interburst_distributions_{}{}.png'.format(
                        string,
                        k,
                        independent_var
                    ))
                    plt.clf()
                    plt.close()

    def plot_example_animations(self):
        """Call a simulation's animation functionality."""
        for identifier, simulation_list in self.experiment_results.items():
            for simulation in simulation_list:
                if simulation.use_kuramato:
                    simulation.animate_phase_bins(self.now, show_gif=False, write_gif=True)
            for i, simulation in enumerate(simulation_list):
                #   simulation.animate_walk(self.now, show_gif=False, write_gif=True)
                simulation.plot_bursts(self.now, instance=i, show_gif=False, write_gif=True)

    def compare_time_series(self):
        """
        Call a simulation's plot_bursts functionality for all of the best betas.

        Assumes invoked on data files for 5 best (1 each) beta dicts
        """
        burst_dict = {}
        for identifier, simulation_list in self.experiment_results.items():
            for simulation in simulation_list:
                k = simulation.total_agents
                bursts_at_each_timestep = simulation.get_burst_data()
                if burst_dict.get(k) is not None:
                    burst_dict[k].append(bursts_at_each_timestep)  # time series
                else:
                    burst_dict[k] = [bursts_at_each_timestep]
        plotting_dict = {}
        for agent_count in burst_dict.keys():
            plotting_dict[agent_count] = self.get_parsimonious_timeseries(burst_dict[agent_count])
        fig, ax = plt.subplots()
        for agent_count in plotting_dict.keys():
            ax.plot(list(plotting_dict[agent_count].keys()), list(plotting_dict[agent_count].values()),
                    label='{}ff'.format(agent_count), lw=1, alpha=0.8)
        plt.legend()
        plt.xlabel('T')
        plt.ylabel('Flashes')
        plt.title('Compare time series of best beta, N=[1,5,10,15,20]')
        plt.show()

    @staticmethod
    def get_parsimonious_timeseries(list_of_dicts):
        best_dict = {}
        max_zeroes = 0
        for l in list_of_dicts:
            vals = list(l.values())
            zeroes = vals.count(0)
            if zeroes > max_zeroes:
                best_dict = l
                max_zeroes = zeroes
        return best_dict

    def compare_simulations(self):
        """
        Plot simulations of a subset against simulations of another subset

        (was: obstacles, uses that flag as an indicator).
        """
        name = list(self.experiment_results.keys())[0]
        step_count = self.experiment_results[name][0].steps
        obstacle_simulations = []
        non_obstacle_simulations = []
        value = list(self.experiment_results.values())[0][0]
        num_agents = value.total_agents
        steps = step_count
        show = False
        write = True
        for identifier, simulation_list in self.experiment_results.items():
            for i, simulation in enumerate(simulation_list):
                if simulation.obstacles:
                    obstacle_simulations.append(simulation)
                else:
                    non_obstacle_simulations.append(simulation)
        for i, (sim_1, sim_2) in enumerate(zip(non_obstacle_simulations, obstacle_simulations)):
            bursts_axis = plt.axes(xlim=(7000, steps), ylim=(0, num_agents))
            sim_1.plot_bursts(self.now, i, show_gif=show, write_gif=write, shared_ax=bursts_axis)
            sim_2.plot_bursts(self.now, i, show_gif=show, write_gif=write, shared_ax=bursts_axis)
            plt.xlabel('t')
            plt.ylabel('num flashes')
            plt.legend()
            plt.title('Flashes over time ' + '{}ff'.format(len(sim_1.firefly_array)) + ' with and without theory')
                      #value.boilerplate + )
            if write:
                plt.savefig('data/time_series_comparison_{}.png'.format(i))
                plt.close()
            if show:
                plt.show()
        try:
            assert len(obstacle_simulations) == len(non_obstacle_simulations), \
                'Need both obstacle and non obstacle sims!'
        except AssertionError:
            return

        if show:
            plt.show()
        if write:
            save_string = value.set_save_string('flashplot_combined', self.now)
            save_string = save_string
            plt.savefig(save_string)
            plt.close()

    def plot_mean_vector_length_results(self):
        """Directly plot statistical results from a simulation."""
        name = list(self.experiment_results.keys())[0]
        step_count = self.experiment_results[name][0].steps
        ax = plt.axes(xlim=(0, step_count + 1), ylim=(0, 1.05))
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean resultant vector length')
        for identifier, simulations in self.experiment_results.items():
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
