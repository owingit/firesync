import networkx as nx
import numpy
import matplotlib.pyplot as plt
import math
import random
import csv
import pickle

from bokeh.io import output_file, show
from bokeh.models import Ellipse, GraphRenderer, StaticLayoutProvider
from bokeh.models.graphs import from_networkx
from bokeh.palettes import Spectral8
from bokeh.plotting import figure
import sklearn
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde


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


def get_initial_interburst_interval():
    with open('data/ib01ff.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    good_data = [float(d[0]) for d in data]
    trimmed_data = [d for d in good_data if 120 > d > 3]
    n, x = numpy.histogram(trimmed_data, bins=400, density=True)
    density = gaussian_kde(trimmed_data, bw_method=0.1)
                           #bw_method=gaussian_kde.covariance_factor)
    # plt.plot(x, density(x))
    # plt.xlabel('Tb (s)')
    # plt.ylabel('Freq')
    # plt.xlim([0, 120])
    # plt.title('Smoothed input distribution')
    # plt.show()

    choice = -1
    while choice < 0.0:
        choice = density.resample(size=1)[0][0]
    return choice * 10


def calc_means_stds(interburst_interval_distribution, swarm_interburst_interval_distribution, on_betas=False):
        individual_dicts = [vals for vals in interburst_interval_distribution.values()]
        i_d = {list(individual_dicts[i].keys())[0]: list(individual_dicts[i].values())
               for i in range(len(individual_dicts))}
        keys = i_d.keys()
        individual_means = {k: 0 for k in keys}
        individual_stds = {k: 0 for k in keys}
        for key in keys:
            lvals = [value for list_of_vals in i_d[key] for vals in list_of_vals for value in vals]
            if len(lvals) > 0:
                individual_means[key] = numpy.mean(lvals)
                individual_stds[key] = numpy.std(lvals)
            else:
                individual_means[key] = 'No distribution found'
                individual_stds[key] = 'No distribution found'
        swarm_dicts = [v for v in swarm_interburst_interval_distribution.values()]
        s_d = {list(swarm_dicts[i].keys())[0]: list(swarm_dicts[i].values())
               for i in range(len(swarm_dicts))}
        keys = s_d.keys()
        swarm_means = {k: 0 for k in keys}
        swarm_stds = {k: 0 for k in keys}
        for key in keys:
            lvals = [s_value for s_list_of_vals in s_d[key] for s_vals in s_list_of_vals for s_value in s_vals]
            if len(lvals) > 0:
                swarm_means[key] = numpy.mean(lvals)
                swarm_stds[key] = numpy.std(lvals)
            else:
                swarm_means[key] = 'No distribution found'
                swarm_stds[key] = 'No distribution found'

        return swarm_means, swarm_stds, individual_means, individual_stds
