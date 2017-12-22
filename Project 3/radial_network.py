from k_means import KMeans
from radial_basis_network import RadialBasisNetwork
import numpy as np
import pdb

tests = [('data/computer_data.txt', -1),
         ('data/fire_data.txt', -1)]

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
    total_performance = 0.0
    for holdout_fold_idx in range(5):
        rbn = RadialBasisNetwork(0.002, learner_type)
        # Train the network
        rbn.train(data_instances[ \
                    np.array( \
                        np.setdiff1d(data_indices, data_indices[ \
                                fold_size * holdout_fold_idx : \
                                fold_size * holdout_fold_idx + fold_size]))], False)

        # Predict test instances
        predictions = rbn.predict(
                data_instances[np.array(
                    data_indices[
                        fold_size * holdout_fold_idx : 
                        fold_size * holdout_fold_idx + fold_size])])
        total_performance += \
                sum((data_instances[np.array(
                    data_indices[
                        fold_size * holdout_fold_idx : 
                        fold_size * holdout_fold_idx + fold_size]),-1] - np.array(predictions)) ** 2)
    total_performance = total_performance / 5
    print("Ave mean squared error: %f\n" % total_performance)
