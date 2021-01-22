import networkx as nx
import numpy
import matplotlib.pyplot as plt
import math
import random
import csv

from bokeh.io import output_file, show
from bokeh.models import Ellipse, GraphRenderer, StaticLayoutProvider
from bokeh.models.graphs import from_networkx
from bokeh.palettes import Spectral8
from bokeh.plotting import figure

TSTAR_RANGE = 100


def get_uniform_coordinates(i, side_length, total):
    """Distribute a firefly at index i within a uniform distribution."""
    positionsx = numpy.linspace(0, side_length - (side_length / math.ceil(math.sqrt(total)) + 1),
                                math.ceil(math.sqrt(total)))
    positionsy = numpy.linspace(0, side_length - (side_length / math.ceil(math.sqrt(total)) + 1),
                                math.ceil(math.sqrt(total)))
    x, y = numpy.meshgrid(positionsx, positionsy)
    x_coords = x.flatten()
    y_coords = y.flatten()
    return x_coords[i], y_coords[i]


def test_initial_coordinates():
    """Plot uniform distribution for visual confirmation."""
    total = 150
    side_length = 15
    for i in range(0, total):
        plt.scatter(get_uniform_coordinates(i, side_length, total)[0],
                    get_uniform_coordinates(i, side_length, total)[1])
    plt.show()


def get_initial_direction(theta_star_range):
    """Set up a direction from within a range of angles."""
    all_directions = numpy.linspace(-math.pi, math.pi, theta_star_range)
    return all_directions[random.randint(0, theta_star_range - 1)]


def generate_line_points(pointa, pointb, num_points):
    """"
    Return a list of nb_points equally spaced points
    between p1 and p2
    """
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (pointb[0] - pointa[0]) / (num_points + 1)
    y_spacing = (pointb[1] - pointa[1]) / (num_points + 1)

    return [[pointa[0] + i * x_spacing, pointa[1] + i * y_spacing] for i in range(1, num_points + 1)]


def centroid(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    _len = len(points)
    if _len > 0:
        centroid_x = sum(x_coords) / _len
        centroid_y = sum(y_coords) / _len
        return [centroid_x, centroid_y]
    else:
        return None


def _visualize(network, i_s, side_length):
    node_indices = (list(network.nodes()))

    x = [network.nodes[node]['xypositions'][0] for node in network.nodes()]
    y = [network.nodes[node]['xypositions'][1] for node in network.nodes()]

    graph_layout = dict(zip(node_indices, zip(x, y)))
    graph = from_networkx(network, layout_function=graph_layout)
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
    plot = figure(title='Network embedding, t={}-{}'.format(min(i_s), max(i_s)), x_range=(0, side_length),
                  y_range=(0, side_length),
                  tools='', toolbar_location=None)
    plot.renderers.append(graph)

    output_file('graph.html')
    show(plot)
    plt.xlim(0, side_length)
    plt.ylim(0, side_length)
    d = dict(network.out_degree)
    node_size = [(v+1) * 20 for v in d.values()]

    edge_attrs = nx.get_edge_attributes(network, 'timestep_of_edge')
    edge_color_range = list(range(len(list(edge_attrs.values()))))
    edge_options = {
        "edge_color": list(edge_color_range),
        "width": 2,
        "edge_cmap": plt.cm.cool,
        "with_labels": False,
    }
    nx.draw(network, graph_layout, node_size=node_size, **edge_options)
    plt.title('Network embedding, t={}-{}'.format(min(i_s), max(i_s)))

    plt.show()


def cluster_indices(label, labels):
    return numpy.where(labels == label)[0]


def get_initial_distribution():
    with open('data/ib01ff.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    good_data = [float(d[0]) for d in data]
    trimmed_data = [(d / 10) for d in good_data if d > 3.0]
    print(trimmed_data)
    print('close')
