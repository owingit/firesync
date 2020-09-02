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
import bresenham
from matplotlib.animation import FuncAnimation
import collections


class Simulation:
    def __init__(self, num_agents, side_length, step_count, thetastar, coupling_strength, Tb, r_or_u="uniform",
                 use_obstacles=False):
        self.total_agents = num_agents
        self.n = side_length
        self.coupling_strength = coupling_strength
        self.Tb = Tb
        self.steps = step_count
        self.firefly_array = []
        self.r_or_u = r_or_u
        self.tstar_seed = thetastar
        thetastars = [np.linspace(-thetastar, thetastar, simulation_helpers.TSTAR_RANGE)]
        self.thetastar = list(thetastars[random.randint(0, len(thetastars) - 1)])
        self.has_run = False
        self.obstacles = None
        self.use_obstacles = use_obstacles
        self.init_obstacles()

        for i in range(0, self.total_agents):
            self.firefly_array.append(Firefly.Firefly(
                i, total=self.total_agents, tstar=self.thetastar,
                tstar_range=simulation_helpers.TSTAR_RANGE,
                n=self.n, steps=self.steps, r_or_u=self.r_or_u,
                use_periodic_boundary_conditions=False,
                tb=self.Tb, obstacles=self.obstacles)
            )

        self.num_fireflies_with_phase_x = collections.OrderedDict()
        self.mean_resultant_vector_length = collections.OrderedDict()
        self.init_stats()
        self.boilerplate = '({}density, {}rad natural frequency)'.format(self.total_agents / (self.n*self.n), self.Tb)

    def init_obstacles(self):
        num_obstacles = random.randint(10, 20)
        if self.use_obstacles is True:
            self.obstacles = obstacle.ObstacleGenerator(num_obstacles, self.n)

    def init_stats(self):
        for i in range(self.steps):
            self.num_fireflies_with_phase_x[i] = {key: 0 for key in range(0, 360)}
        for firefly in self.firefly_array:
            phase_in_degrees = int(math.degrees(firefly.phase[0]))
            if phase_in_degrees < 0:
                phase_in_degrees += 360
            self.num_fireflies_with_phase_x[0][phase_in_degrees] += 1

    def run(self):
        for step in range(1, self.steps):
            for firefly in self.firefly_array:
                firefly.move(step, self.obstacles)
            self.kuramato_phase_interactions(step)
            ff_phases = [ff.phase[step] for ff in self.firefly_array]
            mean_resultant_vector_length = self.circ_r(np.array(ff_phases))
            self.mean_resultant_vector_length[step] = float(mean_resultant_vector_length)

        self.has_run = True

    def kuramato_phase_interactions(self, step):
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
                        line = list(bresenham.bresenham(int(ff_i.positionx[step]), int(ff_i.positiony[step]),
                                                        int(ff_j.positionx[step]), int(ff_j.positiony[step])))
                        for obstacle in self.obstacles.obstacle_array:
                            if not skip_dist:
                                for xy in line:
                                    if obstacle.contains(xy[0], xy[1]):
                                        skip_dist = True
                                        break
                        if skip_dist is True:
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
                             interval=25, blit=False, repeat=False)
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
        assert self.has_run, "Animation cannot render until the simulation has been run!"
        plt.style.use('seaborn-pastel')
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.n), ylim=(0, self.n))
        color_dict = self.setup_color_legend(ax)

        xdatas = {n: [] for n in range(0, self.total_agents)}
        ydatas = {n: [] for n in range(0, self.total_agents)}

        firefly_paths = [ax.plot([], [], '*')[0] for _ in self.firefly_array]

        # TODO: set color by phase
        def animate(i, flies, lines):
            for line, fly in zip(lines, flies):
                xdatas[fly.number].append(fly.trace.get(i)[0])
                ydatas[fly.number].append(fly.trace.get(i)[1])
                line.set_data(xdatas[fly.number][0], ydatas[fly.number][0])
                deg = math.degrees(fly.phase[i])
                if deg < 0:
                    deg += 360
                line.set_color(color_dict[int(deg)])
                xdatas[fly.number].pop(0)
                ydatas[fly.number].pop(0)
            ax.set_title('2D Walk Phase Interactions (step {})'.format(i) + self.boilerplate)
            return lines

        ax.set_xlim([0.0, self.n])
        ax.set_xlabel('X')

        ax.set_ylim([0.0, self.n])
        ax.set_ylabel('Y')
        if self.obstacles:
            for obstacle in self.obstacles.obstacle_array:
                ax.add_artist(plt.Circle((obstacle.centerx, obstacle.centery), obstacle.radius))

        anim = FuncAnimation(fig, animate, frames=self.steps, fargs=(self.firefly_array, firefly_paths),
                             interval=100, blit=False)
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

    @staticmethod
    def setup_color_legend(axis):
        steps = 360
        color_dict = {}
        hsv = matplotlib.cm.get_cmap('hsv', steps)
        norm = matplotlib.colors.Normalize(0, 360)
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

        cmap = matplotlib.colors.ListedColormap(hsv(np.tile(np.linspace(0, 1, steps), 2)))
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
