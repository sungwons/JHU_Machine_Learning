from genetic_algorithm import *
from hac import HAC
from k_means import KMeans

# This program reads in the test data and runs SFS and GA feature selection using k-means and HAC clustering

# Datasets to test
tests = [('data/glass_data.txt', 7),
         ('data/iris_data.txt', 3),
         ('data/spam_data.txt', 2)]

for test in tests:
   data_instances = []
   data_file = open(test[0])
   print "Running with %s" % test[0]
   for line in data_file:
       line_split = line.split(',')
       data_instances.append(map(float, line_split))
   data_instances = np.array(data_instances)

   # Run SFS using k-means and HAC
   kmeans_model = KMeans(test[1])
   hac_model = HAC(test[1])

   # Glass dataset
   if "glass" in test[0]:
      kmeans_sfs_glass = np.array([1,3])
      kmeans_model.cluster(data_instances[:,kmeans_sfs_glass])
      print("K-means SFS glass performance = %f" % kmeans_model.calculate_performance())

      kmeans_ga_glass = np.array([0,1,2,3,4,5,6])
      kmeans_model = KMeans(test[1])
      kmeans_model.cluster(data_instances[:,kmeans_ga_glass])
      print("K-means GA glass performance = %f" % kmeans_model.calculate_performance())

      hac_sfs_glass = np.array([0])
      hac_model.cluster(data_instances[:,hac_sfs_glass])
      print("HAC SFS glass performance = %f" % hac_model.calculate_performance())

   # Iris dataset
   elif "iris" in test[0]:
      kmeans_sfs_iris = np.array([1])
      kmeans_model = KMeans(test[1])
      kmeans_model.cluster(data_instances[:,kmeans_sfs_iris])
      print("K-means SFS iris performance = %f" % kmeans_model.calculate_performance())

      kmeans_ga_iris = np.array([0,1])
      kmeans_model = KMeans(test[1])
      kmeans_model.cluster(data_instances[:,kmeans_ga_iris])
      print("K-means GA iris performance = %f" % kmeans_model.calculate_performance())

      hac_sfs_iris = np.array([0])
      hac_model.cluster(data_instances[:,hac_sfs_iris])
      print("HAC SFS iris performance = %f" % hac_model.calculate_performance())

      hac_ga_iris = np.array([1,2])
      hac_model = HAC(test[1])
      hac_model.cluster(data_instances[:,hac_ga_iris])
      print("HAC GA iris performance = %f" % hac_model.calculate_performance())

   # spam dataset
   else:
       kmeans_sfs_spam = np.array([0])
       kmeans_model = KMeans(test[1])
       kmeans_model.cluster(data_instances[:,kmeans_sfs_spam])
       print("K-means SFS spam performance = %f" % kmeans_model.calculate_performance())

       kmeans_ga_spam = np.array([2,6,8,9,16,18,19,23,24,25,26,29,30,34,39,40,41,43,44,47,49,50])
       kmeans_model = KMeans(test[1])
       kmeans_model.cluster(data_instances[:,kmeans_ga_spam])
       print("K-means GA spam performance = %f" % kmeans_model.calculate_performance())




