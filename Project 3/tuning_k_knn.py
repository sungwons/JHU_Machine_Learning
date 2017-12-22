from k_means import KMeans
from k_nearest_neighbor import KNearestNeighbor
import numpy as np
import pdb

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

    # 5 fold cross validation
    fold_size = data_instances.shape[0] / 5
    data_indices = [idx for idx in range(data_instances.shape[0])]
    for k in range(1,100,5):
        total_performance = 0.0
        for holdout_fold_idx in range(5):
            #  for classification, vote to determine centroid classification
            #  for regression, average to find centroid estimate
            #  feed centroids into k-NN as training data
            if "ecoli" in test[0] or "image" in test[0]:
                learner_type = "CLASSIFICATION"
            else:
                learner_type = "REGRESSION"
            kNN_model = KNearestNeighbor(k, learner_type)
            kNN_model.train(data_instances[ \
                    np.array( \
                        np.setdiff1d(data_indices, data_indices[ \
                                fold_size * holdout_fold_idx : \
                                fold_size * holdout_fold_idx + fold_size]))])
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
        print("k = %d, score = %f" % (k, ave_performance))
