import numpy as np
import pdb
from feed_forward_network import FeedForwardNetwork

# Data sets
tests = ['data/cancer_data.txt',
    'data/glass_data.txt',
    'data/iris_data.txt',
    'data/soybean_data.txt',
    'data/vote_data.txt']

num_of_outputs = {
    'data/cancer_data.txt': 2,
    'data/glass_data.txt': 7,
    'data/iris_data.txt': 3,
    'data/soybean_data.txt': 4,
    'data/vote_data.txt':  2,
}

learning_rates = {
    'data/cancer_data.txt': 0.2,
    'data/glass_data.txt': 0.2,
    'data/iris_data.txt': 0.02,
    'data/soybean_data.txt': 0.2,
    'data/vote_data.txt':  0.00002,
}

tuned_first_layer_size = {
    'data/cancer_data.txt': 9,
    'data/glass_data.txt': 7,
    'data/iris_data.txt': 10,
    'data/soybean_data.txt': 18,
    'data/vote_data.txt':  5,
}
for test in tests:
    data_instances = []
    data_file = open(test)
    print("Running with %s" % test)
    for line in data_file:
        # Digest read data
        line_split = line.split(',')
        data_instances.append(map(float, line_split))
    data_instances = np.array(data_instances)
    # Normalize continuous attributes
    if 'iris' in test:
        for colm_idx in range(data_instances.shape[1] - 1):
            column = data_instances[:, colm_idx]
            data_instances[:, colm_idx] = (column - np.mean(column)) / (2.0 * np.std(column))
    # Shuffle data instances
    np.random.shuffle(data_instances)

    for num_of_hidden_nodes in [i for i in range(0, 3, 1)]:
        print("Testing with %d hidden nodes" % num_of_hidden_nodes)
        data_indices = [idx for idx in range(data_instances.shape[0])]
        # 5-fold cross validation
        num_of_folds = 5
        fold_size = (data_instances.shape[0]) / num_of_folds
        total_performance = 0.0
        for holdout_fold_idx in range(num_of_folds):
            training_indices = np.array(
                np.setdiff1d(
                    data_indices, 
                    data_indices[fold_size * holdout_fold_idx : \
                                 fold_size * holdout_fold_idx + fold_size]))
            test_indices = np.array([i for i in range(
                fold_size * holdout_fold_idx, fold_size * holdout_fold_idx + fold_size)])
    
            model = FeedForwardNetwork(learning_rates[test], 
                data_instances.shape[1] - 1, 2, [tuned_first_layer_size[test], num_of_hidden_nodes], num_of_outputs[test])
            # Train the model
            model.train(data_instances[training_indices])

            # Test performance on test set
            predictions = model.predict(data_instances[test_indices, :-1])
            total_performance += \
                sum(predictions == data_instances[test_indices, -1]) / \
                float(test_indices.shape[0])
        print("Average overall classification rate: %f" % (total_performance / num_of_folds))
