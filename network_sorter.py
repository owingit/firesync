import sklearn.cluster as skl_cluster
import numpy as np
import simulation_helpers
import networkx as nx


class NetworkSort:
    def __init__(self, obj_array, steps, phrase_duration):
        if len(obj_array) > 1:
            k = int((steps / phrase_duration))

            timesteps= set()
            for o in obj_array:
                important_steps = [step for step in range(0, steps) if o.flashed_at_this_step[step]]
                for fs in important_steps:
                    timesteps.add(fs)

            clusters = skl_cluster.KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                                          precompute_distances='deprecated', verbose=0, random_state=None, copy_x=True,
                                          n_jobs='deprecated', algorithm='auto').fit(
                np.array(sorted(list(timesteps))).reshape(-1, 1))

            timesteps_in_clusters = []
            for i in set(clusters.labels_):
                data_in_clusters = np.array(sorted(list(timesteps))).reshape(-1, 1)[
                    simulation_helpers.cluster_indices(i, clusters.labels_)]
                timesteps_in_clusters.append([d[0] for d in data_in_clusters])

            self.sorted_timesteps_in_clusters = sorted(timesteps_in_clusters)
