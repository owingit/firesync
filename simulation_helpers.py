import numpy
import matplotlib.pyplot as plt
import math
import random


TSTAR_RANGE = 100


def get_uniform_coordinates(index, side_length, total):
    positionsx = numpy.linspace(0, side_length - (side_length / math.ceil(math.sqrt(total)) + 1),
                                math.ceil(math.sqrt(total)))
    positionsy = numpy.linspace(0, side_length - (side_length / math.ceil(math.sqrt(total)) + 1),
                                math.ceil(math.sqrt(total)))
    x, y = numpy.meshgrid(positionsx, positionsy)
    x_coords = x.flatten()
    y_coords = y.flatten()
    return x_coords[index], y_coords[index]


def test_initial_coordinates():
    total = 150
    side_length = 15
    for i in range(0, total):
        print(get_uniform_coordinates(i, side_length, total)[0], get_uniform_coordinates(i, side_length, total)[1])
        plt.scatter(get_uniform_coordinates(i, side_length, total)[0],
                    get_uniform_coordinates(i, side_length, total)[1])
    plt.show()


def get_initial_direction(theta_star_range):
    all_directions = numpy.linspace(-math.pi, math.pi, theta_star_range)
    return all_directions[random.randint(0, theta_star_range - 1)]
