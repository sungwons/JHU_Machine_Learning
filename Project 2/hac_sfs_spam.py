from genetic_algorithm import *
from hac import HAC
from stepwise_forward_selection import perform_SFS_feature_selection

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

   # Run SFS using k-means and HAC
   hac_model = HAC(test[1])
   chosen_features = perform_SFS_feature_selection(hac_model, "HAC", test[1], data_instances)
   print("HAC chosen features: %s" % str(chosen_features))
   pdb.set_trace()
