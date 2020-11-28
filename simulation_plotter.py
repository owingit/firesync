import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random
from scipy.interpolate import make_interp_spline
from scipy.stats import norm


class Plotter:
    def __init__(self, experiment_results, now):
        self.experiment_results = experiment_results
        self.now = now
        name = list(self.experiment_results.keys())[0]
        self.step_count = self.experiment_results[name][0].steps

    def plot_quiet_period_distributions(self):
        distribution = False
        interburst_interval_distribution = {}
        swarm_interburst_interval_distribution = {}
        for identifier, simulation_list in self.experiment_results.items():
            interburst_interval_distribution[identifier] = {}
            swarm_interburst_interval_distribution[identifier] = {}
            for simulation in simulation_list:
                k = simulation.total_agents
                if not interburst_interval_distribution[identifier].get(k):
                    interburst_interval_distribution[identifier][k] = [simulation.calc_interburst_distribution()]
                else:
                    interburst_interval_distribution[identifier][k].append(simulation.calc_interburst_distribution())

                if not swarm_interburst_interval_distribution[identifier].get(k):
                    swarm_interburst_interval_distribution[identifier][k] = [simulation.swarm_interburst_dist()]
                else:
                    swarm_interburst_interval_distribution[identifier][k].append(simulation.swarm_interburst_dist())

        if distribution:
            self._plot_distributions(interburst_interval_distribution,
                                     swarm_interburst_interval_distribution)
        else:
            self._plot_histograms(interburst_interval_distribution, swarm_interburst_interval_distribution)
            self._plot_all_histograms(interburst_interval_distribution, swarm_interburst_interval_distribution)

    @staticmethod
    def _plot_all_histograms(individual, group):
        dicts = [individual, group]
        bin_counts = [5, 10, 15, 20, 25, 30]
        for bin_count in bin_counts:
            for i, d in enumerate(dicts):
                fig, ax = plt.subplots()
                ax.set_xlabel('Interburst interval')
                ax.set_ylabel('Freq count')
                colors = [cm.jet(x) for x in np.linspace(0.0, 1.0, len(d.keys())+1)]
                ax.set_xlim(10, 50)
                identifier_data = {}
                trials = 25
                colorindex = 0
                for identifier, results in d.items():
                    for simulation_agent_count, iid_list in results.items():
                        iids = [x / 10 for iid in iid_list for x in iid]
                        if not identifier_data.get(simulation_agent_count):
                            identifier_data[simulation_agent_count] = iids
                        else:
                            identifier_data[simulation_agent_count].append(iids)

                sorted_dict = {k: identifier_data[k] for k in sorted(identifier_data)}
                for simulation_agent_count, data in sorted_dict.items():

                    xs = [d for d in data]
                    y, bin_edges = np.histogram(xs, bins=bin_count, density=True)
                    ys = [height for height in y]
                    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                    x_nice = np.linspace(min(bin_centers), max(bin_centers), 300)
                    _nice = make_interp_spline(bin_centers, ys)
                    y_nice = _nice(x_nice)
                    y_np = np.asarray(y_nice)
                    low_values_flags = y_np < 0.0  # Where values are low
                    y_np[low_values_flags] = 0.0
                    ax.plot(bin_centers, ys, label='{}_agents_{}_pts'.format(simulation_agent_count, len(xs)),
                            color=colors[colorindex], )
                    colorindex += 1

                if i == 0:
                    if trials > 1:
                        string = '{}_bins_Individual_avg_over_'.format(bin_count) + str(trials)
                    else:
                        string = '{}_bins_Individual_avg'.format(bin_count)
                else:
                    if trials > 1:
                        string = '{}_bins_Swarm_avg_over_'.format(bin_count) + str(trials)
                    else:
                        string = '{}_bins_Swarm_avg'.format(bin_count)
                plt.title('{}_interburst_histograms'.format(string))
                plt.legend()
                plt.savefig('histograms/{}_interburst_histograms_not_smoothed.png'.format(string))
                plt.clf()
                plt.close()

    def _plot_histograms(self, individual, group):
        dicts = [individual, group]
        for i, d in enumerate(dicts):
            for identifier, results in d.items():
                fig, ax = plt.subplots()
                ax.set_xlabel('Interburst interval')
                ax.set_ylabel('Freq count')
                ax.set_xlim(10, 50)
                for simulation_agent_count, iid_list in results.items():
                    iids = [x / 10 for iid in iid_list for x in iid]
                    ax.hist(iids, density=True, bins=10, color='cyan', edgecolor='black')
                    trials = len(list(d.keys()))
                    if i == 0:
                        if trials > 1:
                            string = 'Individual_avg_over_' + str(trials)
                        else:
                            string = 'Individual_avg'
                    else:
                        if trials > 1:
                            string = 'Swarm_avg_over_' + str(trials)
                        else:
                            string = 'Swarm_avg'
                    plt.title('{}_interburst_distributions_{}ff_{}_steps'.format(string, simulation_agent_count,
                                                                                 self.step_count))
                    plt.savefig('histograms/{}_interburst_distributions_{}ff_{}_steps.png'.format(string,
                                                                                                  simulation_agent_count,
                                                                                                  self.step_count))
                    plt.clf()
                    plt.close()

    def _plot_distributions(self, individual, group):
        dicts = [individual, group]
        for i, d in enumerate(dicts):
            fig, ax = plt.subplots()
            ax.set_xlabel('Interburst interval')
            ax.set_ylabel('Freq distribution')
            colors = [cm.jet(x) for x in np.linspace(0.0, 1.0, len(d.keys()))]
            colorindex = 0
            for identifier, results in d.items():
                identifier_data = {}
                for simulation_agent_count, iid_list in results.items():
                    for iid in iid_list:
                        if not identifier_data.get(simulation_agent_count):
                            identifier_data[simulation_agent_count] = [(np.mean(iid), np.std(iid))]
                        else:
                            identifier_data[simulation_agent_count].append((np.mean(iid), np.std(iid)))

                sorted_dict = {k: identifier_data[k] for k in sorted(identifier_data)}
                for simulation_agent_count, data in sorted_dict.items():
                    means = [datum[0] for datum in data]
                    stds = [datum[1] for datum in data]
                    overall_mean = sum(means) / len(means)
                    overall_std = sum(stds) / len(stds)
                    dist = norm(overall_mean, overall_std)
                    values = [value for value in range(int(overall_mean - (3 * overall_std)), int(overall_mean + (3 * overall_std)))]
                    probabilities = [dist.pdf(value) for value in values]
                    ax.plot(values, probabilities, label=str(simulation_agent_count)+'_pdf', color=colors[colorindex])
                    colorindex += 1

            plt.legend()

            trials = len(list(d.keys()))
            if i == 0:
                if trials > 1:
                    string = 'Individual_avg_over_' + str(trials)
                else:
                    string = 'Individual_avg'
            else:
                if trials > 1:
                    string = 'Swarm_avg_over_' + str(trials)
                else:
                    string = 'Swarm_avg'
            plt.title('{}_interburst_distributions_{}steps_{}'.format(string, self.step_count,
                                                                      "distribution"))
            plt.savefig('{}_interburst_distributions_{}steps_{}.png'.format(string, self.step_count,
                                                                            "distribution"))
            plt.clf()
            plt.close()

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
            assert len(obstacle_simulations) == len(non_obstacle_simulations), 'Need both obstacle and non obstacle sims!'
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
