import numpy as np
import pdb
from regression_decision_tree import RegressionDecisionTree

# Data sets
tests = ['data_sets/cpu_data.txt',
         'data_sets/fire_data.txt',
         'data_sets/red_wine_data.txt',
         'data_sets/white_wine_data.txt']

# Attribute information map
#   stores the number of attributes in each data set,
#   their type and possible values.
attr_info_map = {
                 'data_sets/cpu_data.txt':
                     [("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", [])],
                 'data_sets/fire_data.txt':
                     [("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", [])],
                 'data_sets/white_wine_data.txt':
                     [("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", [])],
                 'data_sets/red_wine_data.txt':
                     [("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", [])],
                }

# Read in data
for test in tests:
    data_instances = []
    data_file = open(test)
    print("Running with %s" % test)
    for line in data_file:
        # Digest read data
        line_split = line.split(',')
        data_instances.append(map(float, line_split))
    data_instances = np.array(data_instances)
    # Shuffle data instances
    np.random.shuffle(data_instances)

    # Construct validation set
    data_indices = [idx for idx in range(data_instances.shape[0])]
    for threshold in [30000, 25000, 10000, 1000]:
        print("Testing stopping threshold = %f" % threshold)
        # 5-fold cross validation
        fold_size = (data_instances.shape[0]) / 5
        total_performance = 0.0
        for holdout_fold_idx in range(5):
            # training_indices = data_indices - holdout_fold indices
            print("Fold %d of 5" % (holdout_fold_idx + 1))
            training_indices = np.array(
                np.setdiff1d(
                    data_indices, 
                    data_indices[fold_size * holdout_fold_idx : fold_size * holdout_fold_idx + fold_size]))
            # test_indices = holdout_fold indices
            test_indices = np.array([i for i in range(
                fold_size * holdout_fold_idx, fold_size * holdout_fold_idx + fold_size)])

            # Initialize decision tree
            tree = RegressionDecisionTree(
                attr_info = attr_info_map[test], stopping_threshold = threshold)

            # Train the tree
            tree.train(training_data = data_instances[training_indices],
                       attributes = [i for i in range(data_instances[:,:-1].shape[1])])
            predictions = tree.predict(data_instances[test_indices, :-1])

            # Test performance on test set
            total_performance += \
                (sum(data_instances[test_indices,-1] - predictions) ** 2) / \
                    float(test_indices.shape[0])
        print("Average mean squared error: %f" % (total_performance / 5))
