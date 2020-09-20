import random
import numpy as np
import math
import simulation_helpers


class Firefly:
    #  Number, total number, theta*, thetastar_range, box dimension, number of steps,
    #  starting distribution, whether initially fed, whether to use periodic boundary conditions
    def __init__(self, i, total, tstar, tstar_range,
                 n, steps, r_or_u,
                 use_periodic_boundary_conditions=True,
                 tb=1.57, obstacles=None):
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
        if obstacles:
            r_or_u = "random"
        if r_or_u == "random":
            points_set = False
            if obstacles:
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
                self.positionx[0] = random.randint(0, n)
                self.positiony[0] = random.randint(0, n)
        else:
            uniform_x_position, uniform_y_position = simulation_helpers.get_uniform_coordinates(i, n, total)
            self.positionx[0] = uniform_x_position
            self.positiony[0] = uniform_y_position
        self.direction = np.zeros(steps)
        self.direction[0] = simulation_helpers.get_initial_direction(tstar_range)
        self.direction_set = False
        self.theta_star = tstar

        self.phase = np.zeros(steps)
        self.phase[0] = random.random() * math.pi * 2

        # integrate and fire params
        self.beta = 0.1
        self.charging_time = 5
        self.discharging_time = 5
        self.is_charging = random.choice([0, 1])  # or 0
        self.voltage_threshold = 1
        self.voltage_instantaneous = np.zeros(steps)
        self.voltage_instantaneous[0] = random.random()
        self.phrase_duration = 100 #ms
        self.flashes_per_burst = random.randint(5, 8)
        self.flashes_left_in_current_burst = self.flashes_per_burst
        self.quiet_period = self.phrase_duration - (
                (self.charging_time + self.discharging_time) * self.flashes_per_burst
        )
        self.flash_steps = []
        self.bursts = []

        # the total path of a firefly through 2d space
        self.trace = {0: (self.positionx[0], self.positiony[0])}

        try:
            self.nat_frequency = random.vonmisesvariate(tb, 100)
            assert 0.80*tb < self.nat_frequency < 1.20*tb
        except AssertionError:
            self.nat_frequency = tb

    def move(self, current_step, obstacles, flip_direction=False):
        """Move a firefly through 2d space using a correlated 2d random walk."""
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
        """Stage a step for completion."""
        if decrease_velocity:
            self.velocity /= 2
        potential_x_position = self.positionx[current_step - 1] + self.velocity * math.cos(direction)
        potential_y_position = self.positiony[current_step - 1] + self.velocity * math.sin(direction)

        self.direction[current_step] = direction
        self.complete_step(current_step, potential_x_position, potential_y_position, obstacles)

    def complete_step(self, current_step, x, y, obstacles):
        """Complete a step if it does not interfere with an obstacle; recall move otherwise."""
        self.positionx[current_step] = x
        self.positiony[current_step] = y
        self.boundary_conditions(current_step)
        if obstacles:
            for obstacle in obstacles.obstacle_array:
                if obstacle.contains(self.positionx[current_step], self.positiony[current_step]):
                    self.positionx[current_step] = self.positionx[current_step-1]
                    self.positiony[current_step] = self.positiony[current_step-1]
                    self.direction[current_step] = self.direction[current_step-1]
                    self.move(current_step, obstacles, flip_direction=True)
        self.velocity = 1.0
        self.trace[current_step] = (self.positionx[current_step], self.positiony[current_step])

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

        if flip_direction:
            self.direction[current_step] = -self.direction[current_step]

    def flash(self, step):
        self.flashes_left_in_current_burst -= 1
        if self.flashes_left_in_current_burst == 0:
            self.flashes_left_in_current_burst = self.flashes_per_burst
            steps_with_flash = list(range(step - self.flashes_per_burst, step))
            self.bursts = steps_with_flash
            self.flash_steps.append(steps_with_flash)
            print(self.name, steps_with_flash)
