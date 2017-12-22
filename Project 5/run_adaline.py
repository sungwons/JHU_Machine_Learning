import numpy as np
import pdb
from adaline import Adaline

# Data sets

tests = ['data/cancer_data_NB_log.txt',
         'data/iris_data_normalized_1.txt',
         'data/iris_data_normalized_2.txt',
         'data/iris_data_normalized_3.txt',
         'data/soybean_data_NB_1.txt',
         'data/soybean_data_NB_2.txt',
         'data/soybean_data_NB_3.txt',
         'data/soybean_data_NB_4.txt',
         'data/vote_data_NB_log.txt']
'''
tests= ['data/glass_data_1.txt',
        'data/glass_data_2.txt',
        'data/glass_data_3.txt',
        'data/glass_data_4.txt',
        'data/glass_data_5.txt',
        'data/glass_data_6.txt',
        'data/glass_data_7.txt']
'''

bias = {
         'data/cancer_data_NB_log.txt': -200.0,
         'data/iris_data_normalized_1.txt': 0.0,
         'data/iris_data_normalized_2.txt': 0.0,
         'data/iris_data_normalized_3.txt': 20.0,
         'data/soybean_data_NB_1.txt': 0.0,
         'data/soybean_data_NB_2.txt': 0.0,
         'data/soybean_data_NB_3.txt': 0.0,
         'data/soybean_data_NB_4.txt': 20.0,
         'data/vote_data_NB_log.txt': 50.0,
         'data/glass_data_1.txt': 0.0,
         'data/glass_data_2.txt': 0.0,
         'data/glass_data_3.txt': 0.0,
         'data/glass_data_4.txt': 0.0,
         'data/glass_data_5.txt': 0.0,
         'data/glass_data_6.txt': 0.0,
         'data/glass_data_7.txt': 0.0
       }

learning_rates = {
    'data/cancer_data_NB_log.txt': 0.2,
    'data/iris_data_normalized_1.txt': 0.02,
    'data/iris_data_normalized_2.txt': 0.2,
    'data/iris_data_normalized_3.txt': 0.00002,
    'data/soybean_data_NB_1.txt': 0.2,
    'data/soybean_data_NB_2.txt': 0.2,
    'data/soybean_data_NB_3.txt': 0.2,
    'data/soybean_data_NB_4.txt': 0.0002,
    'data/vote_data_NB_log.txt':  0.00002,
    'data/glass_data_1.txt': 0.2,
    'data/glass_data_2.txt': 0.2,
    'data/glass_data_3.txt': 0.2,
    'data/glass_data_4.txt': 0.2,
    'data/glass_data_5.txt': 0.2,
    'data/glass_data_6.txt': 0.2,
    'data/glass_data_7.txt': 0.2
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
        for column in data_instances.T:
            column = (column - np.mean(column)) / (2.0 * np.std(column))
    # Shuffle data instances
    np.random.shuffle(data_instances)

    data_indices = [idx for idx in range(data_instances.shape[0])]
    # 10-fold cross validation
    fold_size = (data_instances.shape[0]) / 10
    total_performance = 0.0
    for holdout_fold_idx in range(10):
        print("Cross validation fold %d" % (holdout_fold_idx + 1))
        # training_indices = data_indices - holdout_fold indices
        training_indices = np.array( 
            np.setdiff1d(
                data_indices, 
                data_indices[fold_size * holdout_fold_idx : \
                             fold_size * holdout_fold_idx + fold_size]))
        # test_indices = holdout_fold indices
        test_indices = np.array([i for i in range(
            fold_size * holdout_fold_idx, fold_size * holdout_fold_idx + fold_size)])

        model = Adaline(bias[test], learning_rates[test])
        # Train the model
        model.train(data_instances[training_indices])
        # Test performance on test set
        predictions = model.predict(data_instances[test_indices, :-1])
        total_performance += \
            sum(predictions == data_instances[test_indices, -1]) / \
            float(test_indices.shape[0])
    print("Average overall classification rate: %f" % (total_performance / 10))
