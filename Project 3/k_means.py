import random
import numpy as np
from math import sqrt
    
# This class implements a simple k-means clustering algorithm.
class KMeans:
    def __init__(self, num_of_means):
        #Class constructor
        self.num_of_means = num_of_means
        self.means = []
        self.clusters = {}

    def cluster(self, data):
        # This method calculates the num_of_means centroids for the inputted data.
        self.means = []
        for mean_idx in range(self.num_of_means):
            mean = []
            for col_idx in range(data.shape[1] - 1):
                mean.append(random.uniform(np.min(data[:,col_idx]), np.max(data[:,col_idx])))
            self.means.append(mean)

        means_updating = True
        loops = 0
        while means_updating:
            # Initialize clusters for the new run
            self.clusters = {}
            for init in range(self.num_of_means):
                self.clusters[init] = []

            # For every instance, assign it to the cluster is it closest to
            for instance in data:
                min_distance = float("inf")
                assigned_cluster = -1
                for mean_idx in xrange(len(self.means)):
                    distance = 0.0
                    for attribute_val, mean_val in zip(instance, self.means[mean_idx]):
                        distance += (attribute_val - mean_val) ** 2
                    distance = sqrt(distance)
                    if (distance < min_distance):
                        min_distance = distance
                        assigned_cluster = mean_idx
                self.clusters[assigned_cluster].append(instance)
            # Recalculate means by cycling
            means_updating = False
            for mean_idx in xrange(len(self.means)):
                for attr_idx in xrange(data.shape[1] - 1):
                    old_mean = self.means[mean_idx][attr_idx]
                    if len(self.clusters[mean_idx]) > 0:
                        self.means[mean_idx][attr_idx] = \
                            np.mean(np.array(self.clusters[mean_idx])[:,attr_idx])
                        if abs(old_mean - self.means[mean_idx][attr_idx]) > 0.001:
                            means_updating = True
            loops += 1
        return self.clusters

    def get_centroids(self):
        return self.means

    def calculate_performance(self):
        # This method calculates HAC performance based the a LDA-like objective function.
        return self.calculate_sum_of_centriod_distances() / self.calculate_inner_class_scatter()

    def calculate_sum_of_centriod_distances(self):
        # This method calculates the sum of differences squared between centroids for all centroids.
        total = 0.0
        for mean_1 in self.means:
            for mean_2 in self.means:
                if not mean_1 == mean_2:
                    euclidean_dist = sqrt(sum((mean1 - mean2)**2))
                    total += euclidean_dist ** 2
        return total 

    def calculate_inner_class_scatter(self):
        # This method calculates the distance between each instance in a cluster and the cluster's centroid.
        total = 0.0
        for mean_idx in range(len(self.means)):
            for instance in self.clusters[mean_idx]:
                euclidean_dist = sqrt(sum((instance - self.means[mean_idx])**2))
                total += euclidean_dist ** 2
        return total
