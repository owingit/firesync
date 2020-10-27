import matplotlib.pyplot as plt
import math
import os
import sys

import collections

import networkx as nx


class NetworkAnalyzer:
    def __init__(self, path_to_networks):
        self.without_obstacle_path = 'data/raw_experiment_results/{}'.format(path_to_networks) + '/with_obstacles'
        self.with_obstacle_path = 'data/raw_experiment_results/{}'.format(path_to_networks) + '/without_obstacles'
        self.network_dict = self.parse_network_files()

    def parse_network_files(self):
        organized_networks = {'with_obstacles': self.read_from_path(self.with_obstacle_path),
                              'without_obstacles': self.read_from_path(self.without_obstacle_path)}
        return organized_networks

    def read_from_path(self, path):
        d = {}
        network_files = [nf for nf in os.listdir(path)]
        if network_files:
            for file in network_files:
                cascade_number = int(file.split('cascade_')[1].split('_')[0])
                incidence_in_cascade = int(file.split('network_')[1].split('.')[0])
                actual_network = nx.read_gpickle(path + '/' + file)[0]
                if d.get(cascade_number):
                    d[cascade_number].append((actual_network, incidence_in_cascade))
                else:
                    d[cascade_number] = [(actual_network, incidence_in_cascade)]
            for cascade in d:
                d[cascade] = sorted(d.get(cascade), key=lambda x: x[1])
            d = dict(sorted(d.items()))
        else:
            print('No networks found in path {}'.format(path))
        return d

    def show_cascades(self):
        for simulation_class, cascade_dict in self.network_dict.items():
            for cascade_index, networks_in_cascade in cascade_dict.items():
                for nw, nw_index in networks_in_cascade:
                    plt.title('Cascade {} network {}, {}'.format(cascade_index, nw_index, simulation_class))
                    nx.draw(nw)
                    plt.show()
                    plt.clf()


if __name__ == "__main__":
    network_analyzer = None
    if len(sys.argv) > 1:
        network_analyzer = NetworkAnalyzer(sys.argv[1])
        network_analyzer.show_cascades()
    else:
        print('Network Analyzer requires a networks folder to run!')