import numpy as np
import pdb
from classification_decision_tree import ClassificationDecisionTree

# Data sets to use
tests = ["data_sets/abalone_data.txt",
         "data_sets/car_data.txt",
         "data_sets/image_data.txt"]

# Attribute information map
#   stores the number of attributes in each data set,
#   their type and possible values.
attr_info_map = {"data_sets/test_data.txt":
                     [("DISCRETE", [0, 1, 2]),
                      ("DISCRETE", [0, 1, 2]),
                      ("DISCRETE", [0, 1]),
                      ("DISCRETE", [0, 1])],
                 "data_sets/abalone_data.txt":
                     [("DISCRETE", [0, 1, 2]),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", []),
                      ("CONTINUOUS", [])],
                 "data_sets/car_data.txt":
                     [("DISCRETE", [0, 1, 2, 3, 4]),
                      ("DISCRETE", [0, 1, 2, 3, 4]),
                      ("DISCRETE", [2, 3, 4, 5]),
                      ("DISCRETE", [2, 4, 5]),
                      ("DISCRETE", [0, 1, 2]),
                      ("DISCRETE", [0, 1, 2])],
                 "data_sets/image_data.txt":
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

# Class labels for each data set
class_labels_map = {
                    "data_sets/abalone_data.txt": [i for i in range(0,29)],
                    "data_sets/car_data.txt": [0, 1, 2, 3],
                    "data_sets/image_data.txt": [0, 1, 2, 3, 4, 5, 6]
                   }
# Read in data
for test in tests:
    data_instances = []
    data_file = open(test)
    print("Running with %s" % test)
    # Digest read data
    for line in data_file:
        line_split = line.split(",")
        data_instances.append(map(float, line_split))
    data_instances = np.array(data_instances)
    # Shuffle data instances
    np.random.shuffle(data_instances)

    # Construct validation set
    data_indices = [idx for idx in range(data_instances.shape[0])]
    validation_indices = data_indices[:int(data_instances.shape[0] * 0.10)]
    validation_instances = data_instances[validation_indices]
    # Remove validation instances from data set
    data_instances = data_instances[np.setdiff1d(data_indices, validation_indices)]
    data_indices = [idx for idx in range(data_instances.shape[0])]
    # 5-fold cross validation
    fold_size = (data_instances.shape[0]) / 5
    total_performance = 0.0
    for holdout_fold_idx in range(5):
        # training_indices = data_indices - holdout_fold indices
        training_indices = np.array(
            np.setdiff1d(
                data_indices, 
                data_indices[fold_size * holdout_fold_idx : fold_size * holdout_fold_idx + fold_size]))
        # test_indices = holdout_fold indices
        test_indices = np.array([i for i in range(
            fold_size * holdout_fold_idx, fold_size * holdout_fold_idx + fold_size)])

        # Initialize decision tree
        tree = ClassificationDecisionTree(attr_info_map[test], class_labels=class_labels_map[test])
        # Train the tree
        tree.train(training_data = data_instances[training_indices],
                   attributes = [i for i in range(data_instances[:,:-1].shape[1])])
        # Perform reduced-error pruning with the validation set
        tree.prune(validation_instances)
        # Test performance on test set
        predictions = tree.predict(data_instances[test_indices, :-1])
        total_performance += \
            (100 * (sum(data_instances[test_indices,-1] == predictions) / float(data_instances.shape[0])))
    print("Average accuracy: %f" % (total_performance / 5))
