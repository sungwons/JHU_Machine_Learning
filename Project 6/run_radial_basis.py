import numpy as np
import pdb
from radial_basis_network import RadialBasisNetwork

# Data sets
tests=['data/cancer_data.txt',
       'data/glass_data.txt',
       'data/iris_data.txt',
       'data/soybean_data.txt',
       'data/vote_data.txt']

learning_rates = {
    'data/cancer_data.txt': 0.2,
    'data/glass_data.txt': 0.2,
    'data/iris_data.txt': 0.02,
    'data/soybean_data.txt': 0.2,
    'data/vote_data.txt':  0.00002,
}

spreads = {
    'data/cancer_data.txt': 1,
    'data/glass_data.txt': 0.01,
    'data/iris_data.txt': 1,
    'data/soybean_data.txt': 0.1,
    'data/vote_data.txt':  0.01,
}

num_of_outputs = {
    'data/cancer_data.txt': 2,
    'data/glass_data.txt': 7,
    'data/iris_data.txt': 3,
    'data/soybean_data.txt': 4,
    'data/vote_data.txt':  2,
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

    data_indices = [idx for idx in range(data_instances.shape[0])]
    # 5-fold cross validation
    num_of_folds = 5
    fold_size = (data_instances.shape[0]) / num_of_folds
    total_performance = 0.0
    for holdout_fold_idx in range(num_of_folds):
        # training_indices = data_indices - holdout_fold indices
        training_indices = np.array( 
            np.setdiff1d(
                data_indices, 
                data_indices[fold_size * holdout_fold_idx : \
                             fold_size * holdout_fold_idx + fold_size]))
        # test_indices = holdout_fold indices
        test_indices = np.array([i for i in range(
            fold_size * holdout_fold_idx, fold_size * holdout_fold_idx + fold_size)])
    
        model = RadialBasisNetwork(learning_rates[test], spreads[test], num_of_outputs[test])
        # Train the model
        model.train(data_instances[training_indices])
        # Test performance on test set
        predictions = model.predict(data_instances[test_indices, :-1])
        for inst, actual, predicted in zip(data_instances[test_indices].tolist(), data_instances[test_indices, -1], predictions):
            print("For instance %s, the model predicted %s and the actual label was %s" % (inst, predicted, actual))
        total_performance += \
            sum(predictions == data_instances[test_indices, -1]) / \
            float(test_indices.shape[0])
    print("Average overall classification rate: %f" % (total_performance / num_of_folds))
