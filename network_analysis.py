import matplotlib.pyplot as plt
import math
import os
import sys

import collections

from matplotlib import animation
from matplotlib.animation import FuncAnimation

import networkx as nx

SIDE_LENGTH = 25


class NetworkAnalyzer:
    def __init__(self, path_to_networks):
        self.with_obstacle_path = 'data/raw_experiment_results/{}'.format(path_to_networks) + '/with_obstacles'
        self.without_obstacle_path = 'data/raw_experiment_results/{}'.format(path_to_networks) + '/without_obstacles'
        self.network_dict = self.parse_network_files()
        print(self.network_dict)

    def parse_network_files(self):
        cwd = os.getcwd() + '/'
        if os.path.isdir(cwd + self.with_obstacle_path) and os.path.isdir(cwd + self.without_obstacle_path):
            organized_networks = {'with_obstacles': self.read_from_path(self.with_obstacle_path),
                                  'without_obstacles': self.read_from_path(self.without_obstacle_path)}
        elif os.path.isdir(cwd + self.with_obstacle_path):
            organized_networks = {'with_obstacles': self.read_from_path(self.with_obstacle_path)}
        else:
            organized_networks = {'without_obstacles': self.read_from_path(self.without_obstacle_path)}
        return organized_networks

    def read_from_path(self, path):
        acc = 'accumulated'
        sep = 'separated'
        d = {acc: {}, sep: {}}
        network_files = [nf for nf in os.listdir(path)]
        if network_files:
            network_files = [f for f in network_files if '.mp4' not in f]
            for file in network_files:
                if acc in file:
                    cascade_number = int(file.split('cascade_')[1].split('.')[0])
                    actual_network = nx.read_gpickle(path + '/' + file)
                    d[acc][cascade_number] = actual_network
                else:
                    cascade_number = int(file.split('cascade_')[1].split('_')[0])
                    incidence_in_cascade = int(file.split('network_')[1].split('.')[0])
                    actual_network = nx.read_gpickle(path + '/' + file)[0]
                    if d[sep].get(cascade_number):
                        d[sep][cascade_number].append((actual_network, incidence_in_cascade))
                    else:
                        d[sep][cascade_number] = [(actual_network, incidence_in_cascade)]
            for cascade in d[sep]:
                d[sep][cascade] = sorted(d[sep].get(cascade), key=lambda x: x[1])
            d_sep = dict(sorted(d[sep].items()))
            d_acc = dict(sorted(d[acc].items()))
            ret_d = {acc: d_acc, sep: d_sep}
        else:
            print('No networks found in path {}'.format(path))
            ret_d = None

        return ret_d

    def show_cascades(self):
        sep = 'separated'
        acc = 'accumulated'
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_xlim((0, SIDE_LENGTH))
        ax.set_ylim((0, SIDE_LENGTH))
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        def animate(i, nw_list):
            nw = nw_list[i][1]

            try:
                title_string = 'Network embedding, cascade {} network {}, {}'.format(nw_list[i][0],
                                                                                     nw_list[i][2],
                                                                                     simulation_class)
                if nw_list[i][2] == 0:
                    if nw_list[i][0] != 0:
                        ax.clear()
                        ax.set_xlim((0, SIDE_LENGTH))
                        ax.set_ylim((0, SIDE_LENGTH))
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                node_indices = (list(nw.nodes()))
            except IndexError:
                title_string = 'Network embedding, accumulated cascade {} network {}'.format(nw_list[i][0],
                                                                                             simulation_class)
                ax.clear()
                ax.set_xlim((0, SIDE_LENGTH))
                ax.set_ylim((0, SIDE_LENGTH))
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                node_indices = (list(nw.nodes()))
            x = [nw.nodes[node]['xypositions'][0] for node in nw.nodes()]
            y = [nw.nodes[node]['xypositions'][1] for node in nw.nodes()]

            graph_layout = dict(zip(node_indices, zip(x, y)))

            ax.set_title(title_string)

            if 'accumulated' not in title_string:
                nx.draw_networkx(nw, graph_layout, node_size=5, ax=ax, with_labels=False)
            else:
                d = dict(nw.out_degree)
                node_size = [(v + 1) * 20 for v in d.values()]

                edge_attrs = nx.get_edge_attributes(nw, 'timestep_of_edge')
                edge_color_range = list(range(len(list(edge_attrs.values()))))
                edge_options = {
                    "edge_color": list(edge_color_range),
                    "width": 2,
                    "edge_cmap": plt.cm.Blues,
                    "with_labels": False,
                }
                nx.draw(nw, graph_layout, node_size=node_size, **edge_options)

        for simulation_class, cascade_dict in self.network_dict.items():
            list_of_objects_to_animate = []
            for cascade_index, networks_in_cascade in cascade_dict[sep].items():
                for nw, nw_index in networks_in_cascade:
                    list_of_objects_to_animate.append((cascade_index, nw, nw_index))
            anim = FuncAnimation(fig, animate, frames=len(list_of_objects_to_animate),
                                 fargs=[list_of_objects_to_animate],
                                 interval=200, blit=False, repeat=False)
            writervideo = animation.FFMpegWriter(fps=10)
            if simulation_class == 'with_obstacles':
                save_string = self.with_obstacle_path + '/network_evolution_video.mp4'
            else:
                save_string = self.without_obstacle_path + '/network_evolution_video.mp4'
            anim.save(save_string, writer=writervideo)
        for simulation_class, cascade_dict in self.network_dict.items():
            list_of_objects_to_animate = []
            for cascade_index, accumulated_network in cascade_dict[acc].items():
                list_of_objects_to_animate.append((cascade_index, accumulated_network))
            anim = FuncAnimation(fig, animate, frames=len(list_of_objects_to_animate),
                                 fargs=[list_of_objects_to_animate],
                                 interval=200, blit=False, repeat=False)
            writervideo = animation.FFMpegWriter(fps=10)
            if simulation_class == 'with_obstacles':
                save_string = self.with_obstacle_path + '/fully_accumulated_cascades_video.mp4'
            else:
                save_string = self.without_obstacle_path + '/fully_accumulated_cascades_video.mp4'
            anim.save(save_string, writer=writervideo)

    def extract_interesting_information(self):
        print(self.network_dict)


if __name__ == "__main__":
    network_analyzer = None
    if len(sys.argv) > 1:
        network_analyzer = NetworkAnalyzer(sys.argv[1])
        network_analyzer.show_cascades()
        network_analyzer.extract_interesting_information()
    else:
        print('Network Analyzer requires a networks folder to run!')