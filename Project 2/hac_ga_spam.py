from genetic_algorithm import *
from hac import HAC

# This program reads in the test data and runs SFS and GA feature selection using k-means and HAC clustering

# Datasets to test
tests = [('data/spam_data.txt', 2)]

for test in tests:
   data_instances = []
   data_file = open(test[0])
   print "Running with %s" % test[0]
   for line in data_file:
       line_split = line.split(',')
       data_instances.append(map(float, line_split))
   data_instances = np.array(data_instances)

   # Run GA using k-means 
   hac_model = HAC(test[1])
   chosen_features = perform_GA_feature_selection(hac_model, "HAC", test[1], data_instances)
   feature_set = [idx for idx in xrange(len(chosen_features[0])) if chosen_features[0][idx] == 1]
   print("Chosen features for HAC GA: %s" % str(chosen_features))
   pdb.set_trace()
   for cluster in hac_model.get_clusters():
       print("HAC chosen cluster: %s" % str(cluster))
