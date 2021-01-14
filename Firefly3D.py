import Firefly

import math
import random

import numpy as np

import simulation_helpers


class Firefly3D(Firefly.Firefly):
    def __init__(self, i, total, tstar, tstar_range, n, steps, r_or_u, beta, phrase_duration,
                 use_periodic_boundary_conditions=True, tb=1.57, obstacles=None):
        self.name = "FF #{}".format(i)
        self.number = i
        if use_periodic_boundary_conditions:
            self.boundary_conditions = self.periodic_boundary_conditions
        else:
            self.boundary_conditions = self.non_periodic_boundary_conditions
        self.velocity = 1.0
        self.side_length_of_enclosure = n
        self.positionx = np.zeros(steps)
        self.positiony = np.zeros(steps)
        self.positionz = np.zeros(steps)
        if obstacles:
            r_or_u = "random"
        if r_or_u == "random":
            points_set = False
            if obstacles:
                while not points_set:
                    success = True
                    self.positionx[0] = random.randint(0, n)
                    self.positiony[0] = random.randint(0, n)
                    self.positionz[0] = random.randint(0, n)
                    for obstacle in obstacles:
                        if obstacle.contains(self.positionx[0], self.positiony[0], self.positionz[0]):
                            success = False
                    if success:
                        points_set = True
            else:
                self.positionx[0] = random.randint(0, n)
                self.positiony[0] = random.randint(0, n)
                self.positionz[0] = random.randint(0, n)
        else:
            uniform_x_position, uniform_y_position, uniform_z_position = simulation_helpers.get_uniform_coordinates(
                i, n, total
            )
            self.positionx[0] = uniform_x_position
            self.positiony[0] = uniform_y_position
            self.positionz[0] = uniform_z_position
        self.direction = np.zeros((steps, 2))
        self.direction[0] = simulation_helpers.get_initial_direction(tstar_range, do_3d=True)
        self.direction_set = False
        self.ready = False
        self.theta_star = tstar
        self.timestepsize = 0.1

        self.phase = np.zeros(steps)
        self.phase[0] = random.random() * math.pi * 2

        # integrate and fire params
        self.beta = beta
        self.charging_time = 5
        self.discharging_time = 5  # timesteps, where each timestep = 0.1s
        self.is_charging = 1
        self.voltage_threshold = 1
        self.voltage_instantaneous = np.zeros(steps)
        self.voltage_instantaneous[0] = random.random()
        if phrase_duration == "distribution":
            self.phrase_duration = np.random.uniform(100, 900, size=1)
        else:
            self.phrase_duration = phrase_duration  # timesteps, where each timestep = 0.1s

        self.flashes_per_burst = 7  # random.randint(5, 8)
        self.flashes_left_in_current_burst = self.flashes_per_burst
        self.quiet_period = self.phrase_duration - (
                 (self.charging_time + self.discharging_time) * self.flashes_per_burst
        )
        self.charging_time = self.charging_time * self.timestepsize
        self.discharging_time = self.discharging_time * self.timestepsize
        self.flashed_at_this_step = [False] * steps
        self.ends_of_bursts = []

        # the total path of a firefly through 2d space
        self.trace = {0: (self.positionx[0], self.positiony[0], self.positionz[0])}

        try:
            self.nat_frequency = random.vonmisesvariate(tb, 100)
            assert 0.80*tb < self.nat_frequency < 1.20*tb
        except AssertionError:
            self.nat_frequency = tb

        # x = Math.cos(alpha) * Math.cos(beta)
        # z = Math.sin(alpha) * Math.cos(beta)
        # y = Math.sin(beta)

    def move(self, current_step, obstacles, flip_direction=False):
        """Move a firefly through 2d space using a correlated 2d random walk."""
        step_alpha = self.theta_star[random.randint(0, 99)]
        step_beta = self.theta_star[random.randint(0, 99)]
        decrease_velocity = False
        if current_step == 0:
            direction = self.direction[current_step]
        elif self.direction_set:
            direction = self.direction[current_step - 1]
            self.direction_set = False
        elif flip_direction:
            direction = [(self.direction[current_step - 1][0] * -1) + step_alpha,
                         (self.direction[current_step - 1][1] * -1) + step_beta]

            decrease_velocity = True
        else:
            direction = [self.direction[current_step - 1][0] + step_alpha,
                         self.direction[current_step - 1][1] + step_beta]

        self.attempt_step(current_step, direction, obstacles, decrease_velocity=decrease_velocity)

    def attempt_step(self, current_step, direction, obstacles, decrease_velocity=False):
        """Stage a step for completion."""
        if decrease_velocity:
            self.velocity /= 2
        potential_x_position = self.positionx[current_step - 1] + (
                self.velocity * math.cos(direction[0]) * math.cos(direction[1])
        )
        potential_z_position = self.positionz[current_step - 1] + (
                self.velocity * math.sin(direction[0]) * math.cos(direction[1])
        )
        potential_y_position = self.positiony[current_step - 1] + (
                self.velocity * math.sin(direction[1])
        )

        self.direction[current_step] = direction
        potential_position = (potential_x_position, potential_y_position, potential_z_position)
        self.complete_step(current_step, potential_position, obstacles)

    def complete_step(self, current_step, potential_position, obstacles):
        """Complete a step if it does not interfere with an obstacle; recall move otherwise."""
        self.positionx[current_step] = potential_position[0]
        self.positiony[current_step] = potential_position[1]
        self.positionz[current_step] = potential_position[2]
        self.boundary_conditions(current_step)
        if obstacles:
            for obstacle in obstacles:
                if obstacle.contains(self.positionx[current_step],
                                     self.positiony[current_step],
                                     self.positionz[current_step]):
                    self.positionx[current_step] = self.positionx[current_step-1]
                    self.positiony[current_step] = self.positiony[current_step-1]
                    self.positionz[current_step] = self.positionz[current_step - 1]
                    self.direction[current_step] = self.direction[current_step-1]
                    self.move(current_step, obstacles, flip_direction=True)
        self.velocity = 1.0
        self.trace[current_step] = (self.positionx[current_step],
                                    self.positiony[current_step],
                                    self.positionz[current_step])

    def periodic_boundary_conditions(self, current_step):
        """Going off the edge of the arena returns an agent to the other side."""
        if self.positionx[current_step] > self.side_length_of_enclosure:
            self.positionx[current_step] = self.positionx[current_step] - self.side_length_of_enclosure
        if self.positionx[current_step] < 0:
            self.positionx[current_step] += self.side_length_of_enclosure

        if self.positiony[current_step] > self.side_length_of_enclosure:
            self.positiony[current_step] = self.positiony[current_step] - self.side_length_of_enclosure
        if self.positiony[current_step] < 0:
            self.positiony[current_step] += self.side_length_of_enclosure

        if self.positionz[current_step] > self.side_length_of_enclosure:
            self.positionz[current_step] = self.positionz[current_step] - self.side_length_of_enclosure
        if self.positionz[current_step] < 0:
            self.positionz[current_step] += self.side_length_of_enclosure

    def non_periodic_boundary_conditions(self, current_step):
        """Bounce off the edges of the arena."""
        flip_direction = False
        if self.positionx[current_step] > self.side_length_of_enclosure:
            distance_from_edge = abs(self.positionx[current_step] - self.side_length_of_enclosure)
            self.positionx[current_step] = self.positionx[current_step] - 2 * distance_from_edge
            flip_direction = True
        if self.positionx[current_step] < 0:
            distance_from_edge = abs(0 - self.positionx[current_step])
            self.positionx[current_step] = self.positionx[current_step] + 2 * distance_from_edge
            flip_direction = True

        if self.positiony[current_step] > self.side_length_of_enclosure:
            distance_from_edge = abs(self.positiony[current_step] - self.side_length_of_enclosure)
            self.positiony[current_step] = self.positiony[current_step] - 2 * distance_from_edge
            self.direction[current_step] = -self.direction[current_step]
            flip_direction = True
        if self.positiony[current_step] < 0:
            distance_from_edge = abs(0 - self.positiony[current_step])
            self.positiony[current_step] = self.positiony[current_step] + 2 * distance_from_edge
            flip_direction = True

        if self.positionz[current_step] > self.side_length_of_enclosure:
            distance_from_edge = abs(self.positionz[current_step] - self.side_length_of_enclosure)
            self.positionz[current_step] = self.positionz[current_step] - 2 * distance_from_edge
            self.direction[current_step] = -self.direction[current_step]
            flip_direction = True
        if self.positionz[current_step] < 0:
            distance_from_edge = abs(0 - self.positionz[current_step])
            self.positionz[current_step] = self.positionz[current_step] + 2 * distance_from_edge
            flip_direction = True

        if flip_direction:
            self.direction[current_step] = -self.direction[current_step]
