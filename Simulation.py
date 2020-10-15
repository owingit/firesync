import simulation_helpers
import numpy as np
import itertools
import Firefly
import random
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy import stats
import statistics
import obstacle
import math
from matplotlib.animation import FuncAnimation
import collections


class Simulation:
    def __init__(self, num_agents, side_length, step_count, thetastar, coupling_strength, Tb, r_or_u="uniform",
                 use_obstacles=False, use_kuramato=True):
        self.firefly_array = []
        self.use_kuramato = use_kuramato
        self.use_integrate_and_fire = not self.use_kuramato

        # constants set by run.py
        self.total_agents = num_agents
        self.n = side_length
        self.coupling_strength = coupling_strength
        self.alpha = 2
        self.Tb = Tb
        self.steps = step_count
        self.r_or_u = r_or_u
        self.tstar_seed = thetastar
        thetastars = [np.linspace(-thetastar, thetastar, simulation_helpers.TSTAR_RANGE)]
        self.thetastar = list(thetastars[random.randint(0, len(thetastars) - 1)])
        self.use_obstacles = use_obstacles

        self.has_run = False
        self.obstacles = None
        if self.use_obstacles is True:
            self.init_obstacles()

        # initialize all Firefly agents
        for i in range(0, self.total_agents):
            self.firefly_array.append(Firefly.Firefly(
                i, total=self.total_agents, tstar=self.thetastar,
                tstar_range=simulation_helpers.TSTAR_RANGE,
                n=self.n, steps=self.steps, r_or_u=self.r_or_u,
                use_periodic_boundary_conditions=True,
                tb=self.Tb, obstacles=self.obstacles)
            )
        if self.use_kuramato:
            self.boilerplate = '({}density, {}rad natural frequency)'.format(self.total_agents / (self.n * self.n),
                                                                             self.Tb)
        else:
            self.boilerplate = '{}density, {}beta'.format(self.total_agents / (self.n * self.n),
                                                          self.firefly_array[0].beta)

        # statistics reporting
        self.num_fireflies_with_phase_x = collections.OrderedDict()
        self.mean_resultant_vector_length = collections.OrderedDict()
        self.wave_statistics = collections.OrderedDict()
        self.init_stats()

    def init_obstacles(self):
        """Initialize an array of obstacles randomly placed throughout the arena."""
        num_obstacles = random.randint(10, 20)
        obstacle_generator = obstacle.ObstacleGenerator(num_obstacles, self.n)
        self.obstacles = obstacle_generator.get_obstacles()

    def init_stats(self):
        """Initialize per-timestep dictionaries tracking firefly phase and TODO: more things."""
        for i in range(self.steps):
            self.num_fireflies_with_phase_x[i] = {key: 0 for key in range(0, 360)}
            self.wave_statistics[i] = {}

        phase_zero_fireflies = []
        for firefly in self.firefly_array:
            phase_in_degrees = int(math.degrees(firefly.phase[0]))
            if phase_in_degrees < 0:
                phase_in_degrees += 360
            if 0 <= phase_in_degrees < 1 or 359 < phase_in_degrees <= 360:
                phase_zero_fireflies.append(firefly)
            self.num_fireflies_with_phase_x[0][phase_in_degrees] += 1

        if phase_zero_fireflies:
            phase_zero_fit = np.polyfit([ff.positionx[0] for ff in phase_zero_fireflies],
                                        [ff.positiony[0] for ff in phase_zero_fireflies],
                                        1)
        else:
            phase_zero_fit = [0, 0]
        self.wave_statistics[0]['count'] = len(phase_zero_fireflies)
        self.wave_statistics[0]['regression'] = phase_zero_fit

    def run(self):
        """
        Run the simulation. At each timestep, a firefly moves in relation to obstacles present and
        experiences phase interactions, either by slightly modified Kuramato model interactions or
        TODO: integrate and fire reactions.
        """
        for step in range(1, self.steps):
            print(step)
            for firefly in self.firefly_array:
                firefly.move(step, self.obstacles)
            if self.use_kuramato:
                self.kuramato_phase_interactions(step)
            if self.use_integrate_and_fire:
                self.integrate_and_fire_interactions(step)
            phase_zero_fireflies = [ff
                                    for ff in self.firefly_array
                                    if 0 <= ff.phase[step] < 1 or 359 < ff.phase[step] <= 360
                                    # or ((ff.phase[step-1] + ff.phase[step] - 2 * (360 - ff.phase[step-1])) % 360 <= math.degrees(self.Tb))
                                    ]
            if phase_zero_fireflies:
                phase_zero_fit = np.polyfit(
                    [ff.positionx[step] for ff in phase_zero_fireflies],
                    [ff.positiony[step] for ff in phase_zero_fireflies],
                    1)
            else:
                phase_zero_fit = self.wave_statistics[step-1]['regression']
            self.wave_statistics[step]['count'] = len(phase_zero_fireflies)
            self.wave_statistics[step]['regression'] = phase_zero_fit
            ff_phases = [ff.phase[step] for ff in self.firefly_array]
            if self.use_kuramato:
                mean_resultant_vector_length = self.circ_r(np.array(ff_phases))
            else:
                # ToDO: better metric
                mean_resultant_vector_length = self.circ_r(np.array(
                    [ff.voltage_instantaneous[step] * 2*math.pi for ff in self.firefly_array]))
            self.mean_resultant_vector_length[step] = float(mean_resultant_vector_length)

        self.has_run = True

    def integrate_and_fire_interactions(self, step):
        self.update_epsilon_and_readiness(step)
        self.update_voltages(step)

    def update_voltages(self, step):
        for i in range(0, self.total_agents):
            ff_i = self.firefly_array[i]
            dvt = ff_i.set_dvt(step)

            int_term = 0
            env_signal = 0
            for j in range(0, self.total_agents):
                if i == j:
                    continue
                else:
                    skip_dist = False
                    ff_j = self.firefly_array[j]
                    if self.obstacles:
                        if not skip_dist:

                            line = simulation_helpers.generate_line_points((ff_i.positionx[step], ff_i.positiony[step]),
                                                                           (ff_j.positionx[step], ff_j.positiony[step]),
                                                                           num_points=100)
                            for obstacle in self.obstacles:
                                if not skip_dist:
                                    for xy in line:
                                        if obstacle.contains(xy[0], xy[1]):
                                            skip_dist = True
                                            break
                        if skip_dist:
                            continue
                        else:
                            int_term = int_term - ff_i.beta * (abs(ff_i.is_charging - ff_j.is_charging))
                            env_signal = env_signal + (1 - ff_j.is_charging)
                    else:
                        int_term = int_term - ff_i.beta * (abs(ff_i.is_charging - ff_j.is_charging))
                        env_signal = env_signal + (1 - ff_j.is_charging)

                    # this is what one firefly contributes to the charging / discharging
            ff_i.voltage_instantaneous[step] = ff_i.voltage_instantaneous[step-1] + (dvt + int_term) * 0.1

    def update_epsilon_and_readiness(self, step):
        # update epsilon and readiness
        for i in range(0, self.total_agents):
            ff_i = self.firefly_array[i]

            # update epsilon to discharging (V is high enough)
            if ff_i.voltage_instantaneous[step - 1] >= (2 * ff_i.voltage_threshold / 3):
                if len(ff_i.ends_of_bursts) == 0:
                    # this is the "waited enough time" step
                    ff_i.set_ready()
                elif len(ff_i.ends_of_bursts) > 0 and step - ff_i.ends_of_bursts[-1] > ff_i.quiet_period:
                    ff_i.set_ready()
                if ff_i.ready:
                    ff_i.is_charging = 0

            # update epsilon to charging if agent flashes
            elif ff_i.voltage_instantaneous[step - 1] <= ff_i.voltage_threshold / 3:
                if ff_i.is_charging == 0 and ff_i.ready:
                    ff_i.flash(step)
                    ff_i.is_charging = 1
                if len(ff_i.ends_of_bursts) > 0 and step - ff_i.ends_of_bursts[-1] > ff_i.quiet_period:
                    ff_i.set_ready()
                if len(ff_i.ends_of_bursts) == 0 and not ff_i.flashed_at_this_step[step]:
                    ff_i.set_ready()

    def kuramato_phase_interactions(self, step):
        """Each firefly's phase wave interacts with the phase wave of its detectable neighbors by the Kuramato model."""
        for i in range(0, self.total_agents):
            ff_i = self.firefly_array[i]
            kuramato = 0
            for j in range(0, self.total_agents):
                if i == j:
                    continue
                else:
                    skip_dist = False
                    ff_j = self.firefly_array[j]
                    if self.obstacles:
                        if not skip_dist:

                            line = simulation_helpers.generate_line_points((ff_i.positionx[step], ff_i.positiony[step]),
                                                                           (ff_j.positionx[step], ff_j.positiony[step]),
                                                                           num_points=100)
                            for obstacle in self.obstacles:
                                if not skip_dist:
                                    for xy in line:
                                        if obstacle.contains(xy[0], xy[1]):
                                            skip_dist = True
                                            break
                        if skip_dist:
                            continue

                    dist = ((ff_j.positionx[step] - ff_i.positionx[step]) ** 2 +
                            (ff_j.positiony[step] - ff_i.positiony[step]) ** 2) ** 0.5
                    if dist == 0:
                        continue
                    kuramato_term = math.sin(ff_j.phase[step - 1] - ff_i.phase[step - 1]) / dist
                    kuramato += kuramato_term

            coupling_term = (ff_i.phase[step - 1] + self.coupling_strength * kuramato)
            ff_i.phase[step] = (ff_i.nat_frequency + coupling_term) % math.radians(360)
            phase_key = int(math.degrees(ff_i.phase[step]))
            if phase_key < 0:
                phase_key += 360
            self.num_fireflies_with_phase_x[step][phase_key] += 1

    def animate_phase_bins(self, now, write_gif=False, show_gif=False):
        """Animate the # of ff's in each phase (0 -> 2*pi) over time."""
        assert self.has_run, "Animation cannot render until the simulation has been run!"
        plt.style.use('seaborn-pastel')
        num_bins = 360
        fig = plt.figure()
        ax = plt.axes(xlim=(0, num_bins), ylim=(0, self.total_agents+1))
        ax.set_xlim([0.0, 360])
        ax.set_xlabel('Phase theta in degrees')
        x_formatter = FixedFormatter([
            "0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°", "360°"])
        x_locator = FixedLocator([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax.xaxis.set_major_formatter(x_formatter)
        ax.xaxis.set_major_locator(x_locator)

        ax.set_ylim([0.0, self.total_agents+1])
        ax.set_ylabel('Num agents')
        rects = ax.bar(range(num_bins), self.num_fireflies_with_phase_x[0].values(), align='center', color='blue')

        def animate(i, data):
            for rect, n in zip(rects, data[i].keys()):
                rect.set_height(data[i][n])
            ax.set_title('Num agents with particular phase at step {}'.format(i) + self.boilerplate)

        anim = FuncAnimation(fig, animate, frames=self.steps, fargs=[self.num_fireflies_with_phase_x],
                             interval=50, blit=False, repeat=False)
        save_string = 'data/numphaseovertime_{}agents_{}x{}_k={}_steps={}_{}distribution{}_gif.gif'.format(
                self.total_agents,
                self.n, self.n,
                self.coupling_strength,
                self.steps,
                self.r_or_u,
                now
            )
        if self.use_obstacles:
            save_string = 'data/numphaseovertime_{}agents_{}x{}_k={}_steps={}_{}distribution{}_obstacles.gif'.format(
                self.total_agents,
                self.n, self.n,
                self.coupling_strength,
                self.steps,
                self.r_or_u,
                now
            )
        else:
            save_string = 'data/numphaseovertime_{}agents_{}x{}_k={}_steps={}_{}distribution{}.gif'.format(
                self.total_agents,
                self.n, self.n,
                self.coupling_strength,
                self.steps,
                self.r_or_u,
                now
            )
        if write_gif:
            anim.save(save_string)
        if show_gif:
            plt.show()

    def animate_walk(self, now, write_gif=False, show_gif=False):
        """Animate the 2d correlated random walks of all fireflies, colored by phase."""
        assert self.has_run, "Animation cannot render until the simulation has been run!"
        plt.style.use('seaborn-pastel')
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.n), ylim=(0, self.n))
        if self.use_kuramato:
            color_dict = self.setup_color_legend(ax, self.use_kuramato, self.use_integrate_and_fire)

        xdatas = {n: [] for n in range(0, self.total_agents)}
        ydatas = {n: [] for n in range(0, self.total_agents)}

        firefly_paths = [ax.plot([], [], '*')[0] for _ in self.firefly_array]
        regression_line = ax.plot([], [], color='orange', linewidth=2)[0]

        def animate(i, flies, lines, wavestats, wave):
            for line, fly in zip(lines, flies):
                xdatas[fly.number].append(fly.trace.get(i)[0])
                ydatas[fly.number].append(fly.trace.get(i)[1])
                line.set_data(xdatas[fly.number][0], ydatas[fly.number][0])
                if self.use_kuramato:
                    deg = math.degrees(fly.phase[i])
                    if deg < 0:
                        deg += 360
                    line.set_color(color_dict[int(deg)])
                if self.use_integrate_and_fire:
                    if fly.flashed_at_this_step[i]:
                        line.set_color('red')
                    else:
                        line.set_color('blue')
                    # voltage = fly.voltage_instantaneous[i]
                    # line.set_color(color_dict[int(abs(voltage*100))])
                xdatas[fly.number].pop(0)
                ydatas[fly.number].pop(0)
            if self.use_kuramato:
                all_xs = [fly.positionx for fly in flies if 0 <= fly.phase[i] < 1
                          or 359 < fly.phase[i] <= 360
                          # or ((fly.phase[i - 1] + fly.phase[i] - 2 * (360 - fly.phase[i - 1])) % 360 <= math.degrees(1.57))
                          ]
            else:
                all_xs = [fly.positionx for fly in flies if fly.flashed_at_this_step[i]]
            wave.set_xdata(all_xs)
            wave.set_ydata(wavestats[i]['regression'][0] * np.asarray(all_xs) + wavestats[i]['regression'][1])
            if self.use_kuramato:
                title_str = "Kuramato Model"
            else:
                title_str = "Integrate-and-Fire Model"
            ax.set_title('2D Walk {} Interactions (step {})'.format(title_str, i) + self.boilerplate)
            return lines, wave

        ax.set_xlim([0.0, self.n])
        ax.set_xlabel('X')

        ax.set_ylim([0.0, self.n])
        ax.set_ylabel('Y')
        if self.obstacles:
            for obstacle in self.obstacles:
                ax.add_artist(plt.Circle((obstacle.centerx, obstacle.centery), obstacle.radius))

        interval = 50 if self.use_kuramato else 300
        anim = FuncAnimation(fig, animate, frames=self.steps,
                             fargs=(self.firefly_array, firefly_paths, self.wave_statistics, regression_line),
                             interval=interval, blit=False)
        if self.use_obstacles:
            save_string = 'data/phaseanim_{}agents_{}x{}_k={}_steps={}_{}distribution{}_obstacles.gif'.format(
                self.total_agents,
                self.n, self.n,
                self.coupling_strength,
                self.steps,
                self.r_or_u,
                now
            )
        else:
            save_string = 'data/phaseanim_{}agents_{}x{}_k={}_steps={}_{}distribution{}.gif'.format(
                self.total_agents,
                self.n, self.n,
                self.coupling_strength,
                self.steps,
                self.r_or_u,
                now
            )
        if write_gif:
            anim.save(save_string)
        if show_gif:
            plt.show()

    def plot_bursts(self, now, write_gif=False, show_gif=False):
        """Plot the flash bursts over time"""
        assert self.has_run, "Plot cannot render until the simulation has been run!"
        plt.style.use('seaborn-pastel')
        ax = plt.axes(xlim=(0, self.steps), ylim=(0, self.total_agents))
        bursts_at_each_timestep = self.get_burst_data()
        ax.plot(list(bursts_at_each_timestep.keys()), list(bursts_at_each_timestep.values()))

        ax.set_xlim([0.0, self.steps])
        ax.set_xlabel('Step')

        ax.set_ylim([0.0, self.total_agents])
        ax.set_ylabel('Number of flashers at timestep')
        plt.title('Flashes over time')

        if self.use_obstacles:
            save_string = 'data/flashplot_{}agents_{}x{}_k={}_steps={}_{}distribution{}_obstacles.png'.format(
                self.total_agents,
                self.n, self.n,
                self.coupling_strength,
                self.steps,
                self.r_or_u,
                now
            )
        else:
            save_string = 'data/flashplot_{}agents_{}x{}_k={}_steps={}_{}distribution{}.png'.format(
                self.total_agents,
                self.n, self.n,
                self.coupling_strength,
                self.steps,
                self.r_or_u,
                now
            )
        if write_gif:
            plt.savefig(save_string)
        if show_gif:
            plt.show()


    @staticmethod
    def setup_color_legend(axis, use_kuramato=True, use_integrate_and_fire=False):
        """Set the embedded color axis for the 2d correlated random walk that shows color-phase relations."""
        if use_kuramato:
            steps = 360

            color_dict = {}
            cmap_seed = matplotlib.cm.get_cmap('hsv', steps)
            norm = matplotlib.colors.Normalize(0, steps)
            display_axes = axis.inset_axes(bounds=[0.79, 0.01, 0.21, 0.01])

            cb = matplotlib.colorbar.ColorbarBase(display_axes,
                                                  cmap=matplotlib.cm.get_cmap('hsv', steps),
                                                  norm=norm,
                                                  orientation='horizontal')
            cb.outline.set_visible(False)
            x_formatter = FixedFormatter([
                "0°", "90°", "180°", "270°", "360°"])
            x_locator = FixedLocator([0, 90, 180, 270, 360])
            display_axes.xaxis.tick_top()
            display_axes.tick_params(axis="x", labelsize=6)
            display_axes.set_xlim([0.0, 360.0])
            display_axes.xaxis.set_major_formatter(x_formatter)
            display_axes.xaxis.set_major_locator(x_locator)

            cmap = matplotlib.colors.ListedColormap(cmap_seed(np.tile(np.linspace(0, 1, steps), 2)))
            for i, color in enumerate(cmap.colors):
                color_dict[i] = color
        else:
            steps = 100

            color_dict = {}
            cmap_seed = matplotlib.cm.get_cmap('YlGnBu', steps)
            norm = matplotlib.colors.Normalize(0, steps)
            display_axes = axis.inset_axes(bounds=[0.79, 0.01, 0.21, 0.01])

            cb = matplotlib.colorbar.ColorbarBase(display_axes,
                                                  cmap=matplotlib.cm.get_cmap('YlGnBu', steps),
                                                  norm=norm,
                                                  orientation='horizontal')
            cb.outline.set_visible(False)
            x_formatter = FixedFormatter([
                "0", ".2", ".4", ".6", ".8", "1.0"])
            x_locator = FixedLocator([0, 20, 40, 60, 80, 100])
            display_axes.xaxis.tick_top()
            display_axes.tick_params(axis="x", labelsize=6)
            display_axes.set_xlim([0.0, 100.0])
            display_axes.xaxis.set_major_formatter(x_formatter)
            display_axes.xaxis.set_major_locator(x_locator)

            cmap = matplotlib.colors.ListedColormap(cmap_seed(np.tile(np.linspace(0, 1, steps), 2)))
            for i, color in enumerate(cmap.colors):
                color_dict[i] = color

        return color_dict

    @staticmethod
    def circ_r(alpha, w=None, d=0, axis=0):
        """Computes mean resultant vector length for circular data.

        Args:
            alpha: array
                Sample of angles in radians

        Kwargs:
            w: array, optional, [def: None]
                Number of incidences in case of binned angle data

            d: radians, optional, [def: 0]
                Spacing of bin centers for binned data, if supplied
                correction factor is used to correct for bias in
                estimation of r

            axis: int, optional, [def: 0]
                Compute along this dimension

        Return:
            r: mean resultant length

        Code taken from the Circular Statistics Toolbox for Matlab
        By Philipp Berens, 2009
        Python adaptation by Etienne Combrisson
        """
        if w is None:
            w = np.ones(alpha.shape)
        elif alpha.size is not w.size:
            raise ValueError("Input dimensions do not match")

        # Compute weighted sum of cos and sin of angles:
        r = np.multiply(w, np.exp(1j * alpha)).sum(axis=axis)

        # Obtain length:
        r = np.abs(r) / w.sum(axis=axis)

        # For data with known spacing, apply correction factor to
        # correct for bias in the estimation of r
        if d is not 0:
            c = d / 2 / np.sin(d / 2)
            r = c * r

        return np.array(r)

    def get_burst_data(self):
        to_plot = {i: 0 for i in range(self.steps)}
        for step in range(self.steps):
            for firefly in self.firefly_array:
                if firefly.flashed_at_this_step[step] is True:
                    to_plot[step] += 1
        return to_plot
