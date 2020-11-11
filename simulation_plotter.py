import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class Plotter:
    def __init__(self, experiment_results, now, params):
        self.experiment_results = experiment_results
        self.now = now
        self.params = params

    def plot_quiet_period_distributions(self):
        interburst_interval_distribution = {}
        swarm_interburst_interval_distribution = {}
        for identifier, simulation_list in self.experiment_results.items():
            interburst_interval_distribution[identifier] = {}
            swarm_interburst_interval_distribution[identifier] = {}
            for simulation in simulation_list:
                k = simulation.total_agents
                if not interburst_interval_distribution.get(k):
                    interburst_interval_distribution[identifier][k] = [simulation.calc_interburst_distribution()]
                else:
                    interburst_interval_distribution[identifier][k].append(simulation.calc_interburst_distribution())

                if not swarm_interburst_interval_distribution.get(k):
                    swarm_interburst_interval_distribution[identifier][k] = [simulation.swarm_interburst_dist()]
                else:
                    swarm_interburst_interval_distribution[identifier][k].append(simulation.swarm_interburst_dist())

        self._plot_distributions(interburst_interval_distribution, swarm_interburst_interval_distribution)

    def _plot_distributions(self, individual, group):
        dicts = [individual, group]
        for i, d in enumerate(dicts):
            fig, ax = plt.subplots()
            ax.set_xlabel('Interburst interval')
            ax.set_ylabel('Freq distribution')
            for identifier, results in d.items():
                for simulation_agent_count, iid_list in results.items():
                    means = []
                    overall_std_list = []
                    for iid in iid_list:
                        means.append(np.mean(iid))
                        overall_std_list.append(np.std(iid))

                    overall_mean = sum(means) / len(means)
                    overall_std = sum(overall_std_list) / len(overall_std_list)
                    dist = norm(overall_mean, overall_std)
                    values = [value for value in
                              range(int(overall_mean - (3 * overall_std)), int(overall_mean + (3 * overall_std)))]
                    probabilities = [dist.pdf(value) for value in values]
                    ax.plot(values, probabilities, label=str(simulation_agent_count)+'_pdf')

            plt.legend()
            if i == 0:
                string = 'Individual'
            else:
                string = 'Swarm'
            plt.title('{}_interburst_distributions_{}steps_{}'.format(string, self.params["steps"],
                                                                      self.params["phrases"]))
            plt.savefig('{}_interburst_distributions_{}steps_{}.png'.format(string, self.params["steps"],
                                                                            self.params["phrases"]))
            plt.clf()

    def plot_animations(self):
        """Call a simulation's animation functionality."""
        for identifier, simulation_list in self.experiment_results.items():
            for simulation in simulation_list:
                if simulation.use_kuramato:
                    simulation.animate_phase_bins(self.now, show_gif=False, write_gif=True)
            for simulation in simulation_list:
                simulation.animate_walk(self.now, show_gif=False, write_gif=True)
                simulation.plot_bursts(self.now, show_gif=False, write_gif=True)

    def compare_obstacles_vs_no_obstacles(self):
        """Plot obstacle simulations against no obstacles."""
        obstacle_simulations = []
        non_obstacle_simulations = []
        value = list(self.experiment_results.values())[0][0]
        num_agents = value.total_agents
        steps = value.steps
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

    def plot_mean_vector_length_results(self):
        """Directly plot statistical results from a simulation."""
        ax = plt.axes(xlim=(0, self.params["steps"] + 1), ylim=(0, 1.05))
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
