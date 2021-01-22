import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random
from scipy.interpolate import make_interp_spline
import math
from scipy.stats import norm


class Plotter:
    def __init__(self, experiment_results, now):
        self.experiment_results = experiment_results
        self.now = now
        name = list(self.experiment_results.keys())[0]
        self.step_count = self.experiment_results[name][0].steps

    def plot_quiet_period_distributions(self, on_betas=False):
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
                        ob_swarm_interburst_interval_distribution[identifier][k] = [simulation.swarm_interburst_dist()]
                    else:
                        ob_swarm_interburst_interval_distribution[identifier][k].append(simulation.swarm_interburst_dist())
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
                        swarm_interburst_interval_distribution[identifier][k] = [simulation.swarm_interburst_dist()]
                    else:
                        swarm_interburst_interval_distribution[identifier][k].append(simulation.swarm_interburst_dist())

        if len(ob_interburst_interval_distribution.items()) > 0:
            s_means, s_stds, i_means, i_stds = self.calc_means_stds(ob_interburst_interval_distribution,
                                                                    ob_swarm_interburst_interval_distribution,
                                                                    on_betas=on_betas)
            self._plot_histograms(ob_interburst_interval_distribution,
                                  ob_swarm_interburst_interval_distribution,
                                  on_betas=on_betas)
            # self._plot_all_histograms(ob_interburst_interval_distribution,
            #                           ob_swarm_interburst_interval_distribution,
            #                           obs=True, on_betas=on_betas)
        else:
            s_means, s_stds, i_means, i_stds = self.calc_means_stds(interburst_interval_distribution,
                                                                    swarm_interburst_interval_distribution,
                                                                    on_betas=on_betas)
            self._plot_histograms(interburst_interval_distribution, swarm_interburst_interval_distribution,
                                  on_betas=on_betas)
            # self._plot_all_histograms(interburst_interval_distribution, swarm_interburst_interval_distribution,
            #                           on_betas=on_betas)


    @staticmethod
    def calc_means_stds(interburst_interval_distribution, swarm_interburst_interval_distribution, on_betas=False):
        individual_dicts = [vals for vals in interburst_interval_distribution.values()]
        i_d = {list(individual_dicts[i].keys())[0]: list(individual_dicts[i].values())
               for i in range(len(individual_dicts))}
        keys = i_d.keys()
        individual_means = {k: 0 for k in keys}
        individual_stds = {k: 0 for k in keys}
        for key in keys:
            lvals = [value for list_of_vals in i_d[key] for vals in list_of_vals for value in vals]
            individual_means[key] = np.mean(lvals)
            individual_stds[key] = np.std(lvals)
        swarm_dicts = [v for v in swarm_interburst_interval_distribution.values()]
        s_d = {list(swarm_dicts[i].keys())[0]: list(swarm_dicts[i].values())
               for i in range(len(swarm_dicts))}
        keys = s_d.keys()
        swarm_means = {k: 0 for k in keys}
        swarm_stds = {k: 0 for k in keys}
        for key in keys:
            lvals = [s_value for s_list_of_vals in s_d[key] for s_vals in s_list_of_vals for s_value in s_vals]
            swarm_means[key] = np.mean(lvals)
            swarm_stds[key] = np.std(lvals)
        return swarm_means, swarm_stds, individual_means, individual_stds

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
                ax.set_xlabel('Interburst interval')
                ax.set_ylabel('Freq count')
                colors = [cm.jet(x) for x in np.linspace(0.0, 1.0, len(d.keys())+1)]
                ax.set_xlim(0, 50)
                identifier_data = {}
                colorindex = 0

                for identifier, results in d.items():
                    for k, iid_list in results.items():
                        iids = [x / 10 for iid in iid_list for x in iid]
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
                                color=colors[colorindex], )
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
                plt.savefig('histograms/{}_interburst_histograms_smoothed.png'.format(string))
                plt.clf()
                plt.close()

    def _plot_histograms(self, individual, group, on_betas=False):
        s_means, s_stds, i_means, i_stds = self.calc_means_stds(individual,
                                                                group,
                                                                on_betas=on_betas)
        if not on_betas:
            independent_var = 'ff'
        else:
            independent_var = 'beta_20ff'
        dicts = [individual, group]
        for i, d in enumerate(dicts):
            for identifier, results in d.items():
                fig, ax = plt.subplots()
                ax.set_xlabel('Interburst interval')
                ax.set_ylabel('Freq count')
                ax.set_xlim(0, 50)
                for k, iid_list in results.items():
                    iids = [x / 10 for iid in iid_list for x in iid]
                    ax.hist(iids, density=True, bins=10, color='cyan', edgecolor='black')
                    trials = len(iids)
                    if i == 0:
                        if trials > 1:
                            string = 'Individual_avg_over_' + str(trials) + '_{}mean_{}std'.format(math.floor(i_means[k]),
                                                                                                   math.floor(i_stds[k]))
                        else:
                            string = 'Individual_avg' + '_{}mean_{}std'.format(math.floor(i_means[k]),
                                                                               math.floor(i_stds[k]))
                        if '_obstacles' in identifier:
                            string = 'obs' + string
                    else:
                        if trials > 1:
                            string = 'Swarm_avg_over_' + str(trials) + '_{}mean_{}std'.format(math.floor(s_means[k]),
                                                                                              math.floor(s_stds[k]))
                        else:
                            string = 'Swarm_avg' + '_{}mean_{}std'.format(math.floor(s_means[k]),
                                                                          math.floor(s_stds[k]))
                        if '_obstacles' in identifier:
                            string = 'obs' + string
                    plt.title('{}_interburst_distributions_{}{}_{}_steps'.format(
                        string,
                        k,
                        independent_var,
                        self.step_count,
                        ))
                    plt.savefig('histograms/{}_interburst_distributions_{}{}_{}_steps.png'.format(
                        string,
                        k,
                        independent_var,
                        self.step_count))
                    plt.clf()
                    plt.close()

        min_key = min(s_means, key=lambda q: s_means[q])
        print('Min swarm mean interburst distribution is: ' + str(min(s_means, key=s_means.get)))
        print('std of min mean interburst distribution is: ' + str(s_stds[min_key]))

    def plot_example_animations(self):
        """Call a simulation's animation functionality."""
        for identifier, simulation_list in self.experiment_results.items():
            for simulation in simulation_list:
                if simulation.use_kuramato:
                    simulation.animate_phase_bins(self.now, show_gif=False, write_gif=True)
            for simulation in [random.choice(simulation_list)]:
                simulation.animate_walk(self.now, show_gif=False, write_gif=True)
                simulation.plot_bursts(self.now, show_gif=False, write_gif=True)

    def compare_obstacles_vs_no_obstacles(self):
        """Plot obstacle simulations against no obstacles."""
        obstacle_simulations = []
        non_obstacle_simulations = []
        value = list(self.experiment_results.values())[0][0]
        num_agents = value.total_agents
        steps = self.step_count
        bursts_axis = plt.axes(xlim=(0, steps), ylim=(0, num_agents))
        show = False
        write = True
        for identifier, simulation_list in self.experiment_results.items():
            for simulation in simulation_list:
                if simulation.obstacles:
                    obstacle_simulations.append(simulation)
                else:
                    non_obstacle_simulations.append(simulation)
                simulation.plot_bursts(self.now, show_gif=show, write_gif=write, shared_ax=bursts_axis)
        try:
            assert len(obstacle_simulations) == len(non_obstacle_simulations), \
                'Need both obstacle and non obstacle sims!'
        except AssertionError:
            return
        plt.legend()
        plt.title('Flashes over time ' + value.boilerplate + ' with and without obstacles')
        if show:
            plt.show()
        if write:
            save_string = value.set_save_string('flashplot_combined', self.now)
            save_string = save_string
            plt.savefig(save_string)
            plt.close()

    def plot_mean_vector_length_results(self):
        """Directly plot statistical results from a simulation."""
        ax = plt.axes(xlim=(0, self.step_count + 1), ylim=(0, 1.05))
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
