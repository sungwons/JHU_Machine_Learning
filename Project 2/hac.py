import numpy as np
import pdb

from math import sqrt

'''This class implements hierarchical agglometrive clustering.'''


class HAC:
    def __init__(self, num_of_clusters):
        '''levels_centroids is a list of tuples where each tuple stores the level
        of the clustering and a list of lists representing the clusters at that
        level'''
        self.levels = []
        self.num_of_clusters = num_of_clusters

    def cluster(self, data):
        self.data = data
        ''' Initialize clusters, each instance starts as its
        own cluster'''
        active_clusters = []
        for instance in data:
            active_clusters.append((instance, [instance]))
        num_of_attrs = len(active_clusters[0][0])

        ''' Find the distance between between all possible pair of
        clusters and merge the two that are closest together'''
        while len(active_clusters) > 1:
            # Find the clusters closest together and merge them
            min_distance = float("inf")
            ''' indices_of_min_distance holds the two instance indices
            that are closest together'''
            '''NOTE: Using Euclidean distance'''
            indices_of_min_distance = tuple()
            for cluster_idx_1, cluster_info_1 in enumerate(active_clusters):
                for cluster_idx_2, cluster_info_2 in enumerate(active_clusters):
                    if not cluster_idx_1 == cluster_idx_2:
                        distance = 0.0
                        for attr_idx in xrange(num_of_attrs):
                            distance += (cluster_info_1[0][attr_idx] - cluster_info_2[0][attr_idx]) ** 2
                        distance = sqrt(distance)
                        if (distance < min_distance):
                            min_distance = distance
                            indices_of_min_distance = (cluster_idx_1, cluster_idx_2)
            # Merge closest clusters together
            '''Calculate new cluster's centroid by averaging the values of each attribute
            for the two clusters being merged'''
            new_cluster_centroid = \
                [np.mean((centroid_attr_1, centroid_attr_2)) for centroid_attr_1, centroid_attr_2 in
                 zip(active_clusters[indices_of_min_distance[0]][0], active_clusters[indices_of_min_distance[1]][0])]
            '''Create a list of instances that are contained in this new cluster
            by merging the list of instances that are contained in each cluster
            that is being merged'''
            new_cluster_group = \
                active_clusters[indices_of_min_distance[0]][1] + active_clusters[indices_of_min_distance[1]][1]
            # Remove merged clusters from list of active clusters (clusters yet to be merged)
            active_clusters.pop(max(indices_of_min_distance[0], indices_of_min_distance[1]))
            active_clusters.pop(min(indices_of_min_distance[0], indices_of_min_distance[1]))
            # For readability, represent a cluster as a tuple
            new_cluster = (new_cluster_centroid, new_cluster_group)
            # Add the new cluster to the list of active clusters
            active_clusters.append(new_cluster)
            # Add the current active clusters to a list so we can retrieve this cluster picture later
            # based on the distance between newly merged clusters
            self.levels.append((min_distance, np.array(active_clusters)))
        return self.get_clusters()

    def get_clusters(self):
        '''This method returns num_of_clusters clusters by sorting self.levels based
        on distance between clusters merged at each iteration'''
        self.levels = sorted(self.levels, key=lambda tup: tup[0], reverse=True)
        return self.levels[self.num_of_clusters - 1][1]

    def calculate_performance(self):
        '''This method calculates HAC performance based the a LDA-like objective function.
        Specifically, the sum of the differences between centroids is divided by the
        sum of inner class scatter.'''
        return self.calculate_sum_of_centroid_distances() / self.calculate_inner_class_scatter()

    def calculate_sum_of_centroid_distances(self):
        '''This method calculates the sum of differences squared between centroids for all centroids.'''
        '''NOTE: Euclidean distance is used to convert vector to scalar'''
        total = 0.0
        clusters = self.get_clusters()
        for cluster_1_idx, cluster_1_val in enumerate(clusters):
            for cluster_2_idx, cluster_2_val in enumerate(clusters):
                if not cluster_1_idx == cluster_2_idx:
                    for mean_1, mean_2 in zip(cluster_1_val[0], cluster_2_val[0]):
                        total += (mean_1 - mean_2) ** 2
        return total

    def calculate_inner_class_scatter(self):
        '''This method calculates the distance between each instance in a 
        cluster and the cluster's centroid.'''
        '''NOTE: Euclidean distance is used to convert vector to scalar'''
        total = 0.0
        clusters = self.get_clusters()
        for cluster in clusters:
            for instance in cluster[1]:
                for inst_val, mean_val in zip(instance, cluster[0]):
                    total += (inst_val - mean_val) ** 2
        return total
