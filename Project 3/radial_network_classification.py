from radial_basis_network import RadialBasisNetwork
import numpy as np
import pdb

tests = [('data/image_data_1.txt', 2),
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

best_num_of_means = {
         'data/image_data_1.txt': 5,
         'data/image_data_2.txt': 41,
         'data/image_data_3.txt': 3,
         'data/image_data_4.txt': 19,
         'data/image_data_5.txt': 35,
         'data/image_data_6.txt': 21,
         'data/image_data_7.txt': 19,
         'data/ecoli_data_1.txt': 45,
         'data/ecoli_data_2.txt': 46,
         'data/ecoli_data_3.txt': 5,
         'data/ecoli_data_4.txt': 5,
         'data/ecoli_data_5.txt': 5,
         'data/ecoli_data_6.txt': 5,
         'data/ecoli_data_7.txt': 15,
         'data/ecoli_data_8.txt': 49}

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
    learner_type = "CLASSIFICATION"

    for do_clustering in [True, False]: 
        print("Performing k-means clustering first? %d" % do_clustering)
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
                                    fold_size * holdout_fold_idx + fold_size]))], \
                                    do_clustering, best_num_of_means[test[0]])

            # Predict test instances
            predictions = rbn.predict(
                    data_instances[np.array(
                        data_indices[
                            fold_size * holdout_fold_idx : 
                            fold_size * holdout_fold_idx + fold_size])])
            pdb.set_trace()
            if learner_type == "REGRESSION":
                total_performance += \
                        sum((data_instances[np.array(
                            data_indices[
                                fold_size * holdout_fold_idx : 
                                fold_size * holdout_fold_idx + fold_size]),-1] - np.array(predictions)) ** 2)
            else:
                total_performance += \
                        sum(data_instances[np.array(
                            data_indices[
                                fold_size * holdout_fold_idx : 
                                fold_size * holdout_fold_idx + fold_size]),-1] == np.array(predictions)) / float(fold_size)
                    
        total_performance = total_performance / 5
        print("Ave classification success: %f\n" % total_performance)
