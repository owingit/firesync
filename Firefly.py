import random
import numpy as np
import math
import simulation_helpers

import networkx as nx


class Firefly:
    #  Number, total number, theta*, thetastar_range, box dimension, number of steps,
    #  starting distribution, whether initially fed, whether to use periodic boundary conditions
    def __init__(self, i, total, tstar, tstar_range,
                 n, steps, r_or_u,
                 use_periodic_boundary_conditions=True,
                 Tb=1.57, obstacles=None):
        self.velocity = 1.0
        self.side_length_of_enclosure = n
        self.positionx = np.zeros(steps)
        self.positiony = np.zeros(steps)
        if obstacles:
            r_or_u = "random"
        if r_or_u == "random":
            points_set = False
            while not points_set:
                success = True
                self.positionx[0] = random.randint(0, n)
                self.positiony[0] = random.randint(0, n)
                for obstacle in obstacles.obstacle_array:
                    if obstacle.contains(self.positionx[0], self.positiony[0]):
                        success = False
                if success:
                    points_set = True
        else:
            uniform_x_position, uniform_y_position = simulation_helpers.get_uniform_coordinates(i, n, total)
            self.positionx[0] = uniform_x_position
            self.positiony[0] = uniform_y_position
        self.direction = np.zeros(steps)
        self.direction[0] = simulation_helpers.get_initial_direction(tstar_range)
        self.direction_set = False
        self.theta_star = tstar
        self.trace = {0: (self.positionx[0], self.positiony[0])}
        # natural frequency: 1.57 radians / second
        try:
            self.nat_frequency = random.vonmisesvariate(Tb, 100)
            assert 0.80*Tb < self.nat_frequency < 1.20*Tb
        except AssertionError:
            self.nat_frequency = Tb

        self.name = "FF #{}".format(i)
        self.number = i
        if use_periodic_boundary_conditions:
            self.boundary_conditions = self.periodic_boundary_conditions
        else:
            self.boundary_conditions = self.non_periodic_boundary_conditions

        self.phase = np.zeros(steps)
        self.phase[0] = random.random() * math.pi * 2

    def move(self, current_step, obstacles, flip_direction=False):
        random_int = random.randint(0, 99)
        step_theta = self.theta_star[random_int]
        decrease_velocity = False
        if current_step == 0:
            direction = self.direction[current_step]
        elif self.direction_set:
            direction = self.direction[current_step - 1]
            self.direction_set = False
        elif flip_direction:
            direction = self.direction[current_step - 1] * -1
            direction = direction + step_theta
            decrease_velocity = True
        else:
            direction = self.direction[current_step - 1] + step_theta

        self.attempt_step(current_step, direction, obstacles, decrease_velocity=decrease_velocity)

    def attempt_step(self, current_step, direction, obstacles, decrease_velocity=False):
        if decrease_velocity:
            self.velocity /= 2
        potential_x_position = self.positionx[current_step - 1] + self.velocity * math.cos(direction)
        potential_y_position = self.positiony[current_step - 1] + self.velocity * math.sin(direction)

        self.direction[current_step] = direction
        self.complete_step(current_step, potential_x_position, potential_y_position, obstacles)

    def complete_step(self, current_step, x, y, obstacles):
        self.positionx[current_step] = x
        self.positiony[current_step] = y
        self.boundary_conditions(current_step)
        for obstacle in obstacles.obstacle_array:
            if obstacle.contains(self.positionx[current_step], self.positiony[current_step]):
                self.positionx[current_step] = self.positionx[current_step-1]
                self.positiony[current_step] = self.positiony[current_step-1]
                self.direction[current_step] = self.direction[current_step-1]
                self.move(current_step, obstacles, flip_direction=True)
        self.velocity = 1.0
        self.trace[current_step] = (self.positionx[current_step], self.positiony[current_step])

    def periodic_boundary_conditions(self, current_step):
        if self.positionx[current_step] > self.side_length_of_enclosure:
            self.positionx[current_step] = self.positionx[current_step] - self.side_length_of_enclosure
        if self.positionx[current_step] < 0:
            self.positionx[current_step] += self.side_length_of_enclosure

        if self.positiony[current_step] > self.side_length_of_enclosure:
            self.positiony[current_step] = self.positiony[current_step] - self.side_length_of_enclosure
        if self.positiony[current_step] < 0:
            self.positiony[current_step] += self.side_length_of_enclosure

    def non_periodic_boundary_conditions(self, current_step):
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

        if flip_direction:
            self.direction[current_step] = -self.direction[current_step]
