import simulation_helpers
import numpy as np
import itertools
import Firefly
import random
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FixedLocator, FixedFormatter

import math
import statistics
from matplotlib.animation import FuncAnimation
import collections


class Simulation:
    def __init__(self, num_agents, side_length, step_count, thetastar, coupling_strength, r_or_u="uniform"):
        self.total_agents = num_agents
        self.n = side_length
        self.coupling_strength = coupling_strength
        self.steps = step_count
        self.firefly_array = []
        self.r_or_u = r_or_u
        self.tstar_seed = thetastar
        thetastars = [np.linspace(-thetastar, thetastar, simulation_helpers.TSTAR_RANGE)]
        self.thetastar = list(thetastars[random.randint(0, len(thetastars) - 1)])
        self.has_run = False

        for i in range(0, self.total_agents):
            self.firefly_array.append(Firefly.Firefly(
                i, total=self.total_agents, tstar=self.thetastar,
                tstar_range=simulation_helpers.TSTAR_RANGE,
                n=self.n, steps=self.steps, r_or_u=self.r_or_u,
                use_periodic_boundary_conditions=False)
            )

        self.num_fireflies_with_phase_x = collections.OrderedDict()
        self.init_stats()

    def init_stats(self):
        for i in range(self.steps):
            self.num_fireflies_with_phase_x[i] = {key: 0 for key in range(0, 360)}
        for firefly in self.firefly_array:
            phase_in_degrees = int(math.degrees(firefly.phase[0]))
            self.num_fireflies_with_phase_x[0][phase_in_degrees] += 1

    def run(self):
        num_agents = len(self.firefly_array)
        for step in range(1, self.steps):
            for firefly in self.firefly_array:
                firefly.move(step)
            for i in range(0, num_agents):
                ff_i = self.firefly_array[i]
                for j in range(i, num_agents-1):
                    j += 1
                    ff_j = self.firefly_array[j]
                    dist = ((ff_j.positionx[step] - ff_i.positionx[step]) ** 2 +
                            (ff_j.positiony[step] - ff_i.positiony[step]) ** 2) ** 0.5
                    kuramato = math.sin(ff_j.phase[step-1] - ff_i.phase[step-1]) / dist
                    kuramato_2 = math.sin(ff_i.phase[step-1] - ff_j.phase[step-1]) / dist

                    ff_i.phase[step] = ff_i.phase[step-1] + (self.coupling_strength * kuramato)
                    ff_i.phase[step] = ff_i.phase[step] % math.radians(360)
                    ff_j.phase[step] = ff_j.phase[step-1] + (self.coupling_strength * kuramato_2)
                    ff_j.phase[step] = ff_j.phase[step] % math.radians(360)
                phase_key = int(math.degrees(ff_i.phase[step]))
                self.num_fireflies_with_phase_x[step][phase_key] += 1

        self.has_run = True

    def animate_phase_bins(self, now):
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
            ax.set_title('Num agents with particular phase at step {}'.format(i))

        anim = FuncAnimation(fig, animate, frames=self.steps, fargs=[self.num_fireflies_with_phase_x],
                             interval=25, blit=False, repeat=False)
        # anim.save('data/numphaseovertime_{}agents_{}x{}_k={}_steps={}_{}distribution{}_gif.gif'.format(
        #     self.total_agents,
        #     self.n, self.n,
        #     self.coupling_strength,
        #     self.steps,
        #     self.r_or_u,
        #     now
        # ))
        plt.show()

    def animate_walk(self, now):
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
                line.set_color(color_dict[int(deg)])
                xdatas[fly.number].pop(0)
                ydatas[fly.number].pop(0)
            ax.set_title('2D Walk Phase Interactions (step {})'.format(i))
            return lines

        ax.set_xlim([0.0, self.n])
        ax.set_xlabel('X')

        ax.set_ylim([0.0, self.n])
        ax.set_ylabel('Y')

        anim = FuncAnimation(fig, animate, frames=self.steps, fargs=(self.firefly_array, firefly_paths),
                             interval=100, blit=False)

        anim.save('data/phaseanim_{}agents_{}x{}_k={}_steps={}_{}distribution{}_gif.gif'.format(
            self.total_agents,
            self.n, self.n,
            self.coupling_strength,
            self.steps,
            self.r_or_u,
            now
        ))
        # plt.show()

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
