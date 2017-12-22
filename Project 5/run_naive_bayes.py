import numpy as np
import pdb
from naive_bayes import NaiveBayes

# Data sets
tests = [('data/cancer_data_NB_log.txt', [0,1]),
         ('data/iris_data_NB_1.txt', [0,1]),
         ('data/iris_data_NB_2.txt', [0,1]),
         ('data/iris_data_NB_3.txt', [0,1]),
         ('data/soybean_data_NB_1.txt', [0,1]),
         ('data/soybean_data_NB_2.txt', [0,1]),
         ('data/soybean_data_NB_3.txt', [0,1]),
         ('data/soybean_data_NB_4.txt', [0,1]),
         ('data/vote_data_NB_log.txt', [0,1])]

'''
tests= [('data/glass_data_1.txt', [0,1]),
        ('data/glass_data_2.txt', [0,1]),
        ('data/glass_data_3.txt', [0,1])
        ('data/glass_data_4.txt', [0,1]),
        ('data/glass_data_5.txt', [0,1]),
        ('data/glass_data_6.txt', [0,1]),
        ('data/glass_data_7.txt', [0,1])]
'''

for test in tests:
    data_instances = []
    data_file = open(test[0])
    print("Running with %s" % test[0])
    for line in data_file:
        # Digest read data
        line_split = line.split(',')
        data_instances.append(map(float, line_split))
    data_instances = np.array(data_instances)
    # Shuffle data instances
    np.random.shuffle(data_instances)

    data_indices = [idx for idx in range(data_instances.shape[0])]
    # 10-fold cross validation
    fold_size = (data_instances.shape[0]) / 10
    total_performance = 0.0
    for holdout_fold_idx in range(10):
        # training_indices = data_indices - holdout_fold indices
        training_indices = np.array( 
            np.setdiff1d(
                data_indices, 
                data_indices[fold_size * holdout_fold_idx : \
                             fold_size * holdout_fold_idx + fold_size]))
        # test_indices = holdout_fold indices
        test_indices = np.array([i for i in range(
            fold_size * holdout_fold_idx, fold_size * holdout_fold_idx + fold_size)])

        model = NaiveBayes()
        # Train the model
        model.build_model(data_instances[training_indices])
        
        # Print model
        print("Bayes model = \n%s\n" % model.print_model())

        # Test performance on test set
        predictions = []
        for instance in data_instances[test_indices, :-1]:
            predictions.append(model.predict(instance))
        total_performance += \
            sum(predictions == data_instances[test_indices, -1]) / \
            float(test_indices.shape[0])
    print("Average classification rate: %f" % (total_performance / 10))
