from k_means import KMeans
from k_nearest_neighbor import KNearestNeighbor
import numpy as np


tests = [('data/computer_data.txt', -1),
         ('data/fire_data.txt', -1),
         ('data/image_data_1.txt', 2),
         ('data/image_data_2.txt', 2),
         ('data/image_data_3.txt', 2),
         ('data/image_data_4.txt', 2),
         ('data/image_data_5.txt', 2),
         ('data/image_data_6.txt', 2),
         ('data/image_data_7.txt', 2),
         ('data/ecoli_data_1.txt', 2),
         ('data/ecoli_data_2.txt', 2),
         ('data/ecoli_data_3.txt', 2),
         ('data/ecoli_data_4.txt', 2),
         ('data/ecoli_data_5.txt', 2),
         ('data/ecoli_data_6.txt', 2),
         ('data/ecoli_data_7.txt', 2),
         ('data/ecoli_data_8.txt', 2)]

# These values for k were extracted from the original k-NN experiment
best_ks = {'data/computer_data.txt': 1,
    'data/fire_data.txt': 96,
    'data/image_data_1.txt': 1,
    'data/image_data_2.txt': 1,
    'data/image_data_3.txt': 1,
    'data/image_data_4.txt': 1,
    'data/image_data_5.txt': 1,
    'data/image_data_6.txt': 6,
    'data/image_data_7.txt': 1,
    'data/ecoli_data_1.txt': 6,
    'data/ecoli_data_2.txt': 6,
    'data/ecoli_data_3.txt': 6,
    'data/ecoli_data_4.txt': 6,
    'data/ecoli_data_5.txt': 11,
    'data/ecoli_data_6.txt': 1,
    'data/ecoli_data_7.txt': 1,
    'data/ecoli_data_8.txt': 11 }

# Read in data
for test in tests:
    data_instances = []
    data_file = open(test[0])
    print("Running with %s" % test[0])
    for line in data_file:
        line_split = line.split(',')
        data_instances.append(map(float, line_split))
    data_instances = np.array(data_instances)
    np.random.shuffle(data_instances)
    if "ecoli" in test[0] or "image" in test[0]:
        learner_type = "CLASSIFICATION"
    else:
        learner_type = "REGRESSION"

    # 5 fold cross validation
    fold_size = data_instances.shape[0] / 5
    data_indices = [idx for idx in range(data_instances.shape[0])]
    for num_of_means in range(1,50):
        total_performance = 0.0
        for holdout_fold_idx in range(5):
            # try some num of means
            kmeans_model = KMeans(num_of_means)
            # run k means on training data to find centroids
            clusters = kmeans_model.cluster( \
                data_instances[ \
                    np.array( \
                        np.setdiff1d(data_indices, data_indices[ \
                                fold_size * holdout_fold_idx : \
                                fold_size * holdout_fold_idx + fold_size]))])
            centroids = kmeans_model.get_centroids()
            for cluster_idx in range(len(clusters)):
                ave_label = 0.0
                for instance in clusters[cluster_idx]: 
                    ave_label += instance[-1]
                if len(clusters[cluster_idx]) > 0:
                    ave_label = ave_label / len(clusters[cluster_idx])
                if learner_type == "CLASSIFICATION":
                    ave_label = int(round(ave_label))
                centroids[cluster_idx].append(ave_label)
            #  for classification, vote to determine centroid classification
            #  for regression, average to find centroid estimate
            #  feed centroids into k-NN as training data
            kNN_model = KNearestNeighbor(best_ks[test[0]], learner_type)
            kNN_model.train(centroids)
            #  predict test data using k-NN and average performance
            predictions = kNN_model.predict( \
                data_instances[ \
                    fold_size * holdout_fold_idx : \
                    fold_size * holdout_fold_idx + fold_size])
            if kNN_model.learner_type == "CLASSIFICATION":
                successes = fold_size - \
                    sum(abs(
                        predictions - \
                        data_instances[
                            fold_size * holdout_fold_idx : 
                            fold_size * holdout_fold_idx + fold_size,-1]))
                performance = successes / fold_size
            elif kNN_model.learner_type == "REGRESSION":
                performance = sum((predictions - \
                  data_instances[fold_size * holdout_fold_idx : 
                     fold_size * holdout_fold_idx + fold_size,-1]) ** 2)
            total_performance += performance
        ave_performance = total_performance / 5
        print("num of means = %d, score = %f" % (num_of_means, ave_performance))
