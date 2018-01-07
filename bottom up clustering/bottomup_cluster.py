import numpy as np
from scipy.spatial import distance

class AggloClustering:
    def __init__(self, data):
        self.data = data
        self.clusters = data[:, np.newaxis].tolist()

    def cluster(self, number_of_clusters = 1, linkage = 'single', metric = 'euclidean'):
        print("Computing distance matrix")
        dist_matrix = distance.cdist(self.data, self.data, metric)

        print("Clustering..")

        while len(self.clusters) > number_of_clusters:
            np.fill_diagonal(dist_matrix, np.inf)

            #finding 2 clusters of smallest distance
            i,j = self.find_min_idx(dist_matrix)

            #updating dist matrix for ith row amd column
            dist_matrix[i, :] = self.get_combined_dist_metrics(dist_matrix[i, :],dist_matrix[j, :], i, j, linkage)
            dist_matrix[:, i] = dist_matrix[i, :].T

            #removing jth row and column from dist matrix
            dist_matrix = np.delete(dist_matrix, (j), axis=0)
            dist_matrix = np.delete(dist_matrix, (j), axis=1)

            # combining clusters, moving elements of jth cluster to ith cluster and deleting jth cluster
            self.clusters[i] = np.vstack([self.clusters[i], self.clusters[j]])
            self.clusters.pop(j)

        return self.clusters

    def get_combined_dist_metrics(self, cluster_a_metric, cluster_b_metric,  cluster_a_idx, cluster_b_idx, linkage):
        if linkage == 'single':
            return np.maximum(cluster_a_metric, cluster_b_metric)
        elif linkage == 'complete':
            return np.minimum(cluster_a_metric, cluster_b_metric)
        elif linkage == 'average':
            cluster_a_num = np.array(self.clusters[cluster_a_idx]).shape[0]
            cluster_b_num = np.array(self.clusters[cluster_b_idx]).shape[0]
            return np.add((cluster_a_metric *cluster_a_num),(cluster_b_metric *cluster_b_num))/ (cluster_a_num +cluster_b_num)

    def find_min_idx(self, x):
        return divmod(x.argmin(), x.shape[1])