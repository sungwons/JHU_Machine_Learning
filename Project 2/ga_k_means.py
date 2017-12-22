from genetic_algorithm import *
from hac import HAC
from k_means import KMeans

# This program reads in the test data and runs SFS and GA feature selection using k-means and HAC clustering

# Datasets to test
tests = [('data/spam_data.txt', 2)]

for test in tests:
   data_instances = []
   data_file = open(test[0])
   print("Running with %s" % test[0])
   for line in data_file:
       line_split = line.split(',')
       data_instances.append(map(float, line_split))
   data_instances = np.array(data_instances)

   # Run GA feature selection using k-means and HAC
   kmeans_model = KMeans(test[1])
   hac_model = HAC(test[1])
   chosen_features = perform_GA_feature_selection(kmeans_model, "Kmeans", test[1], data_instances)
   print("Chosen features for K-means GA: %s" % str(chosen_features))
