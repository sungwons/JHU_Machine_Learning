from Winnow import *
from Naive_Bayes import *
import numpy as np

# Datasets to test
files = ['Datasets/cancer_data.txt',
         'Datasets/glass_data_1.txt',
         'Datasets/glass_data_2.txt',
         'Datasets/glass_data_3.txt',
         'Datasets/glass_data_4.txt',
         'Datasets/glass_data_5.txt',
         'Datasets/glass_data_6.txt',
         'Datasets/glass_data_7.txt',
         'Datasets/iris_data_1.txt',
         'Datasets/iris_data_2.txt',
         'Datasets/iris_data_3.txt',
         'Datasets/soybean_data_1.txt',
         'Datasets/soybean_data_2.txt',
         'Datasets/soybean_data_3.txt',
         'Datasets/soybean_data_4.txt',
         'Datasets/voting_data.txt']

for data_file_name in files:
   # Splitting line on ','
   data_instances = []
   data_file = open(data_file_name)
   print("Running with %s" % data_file_name)
   for line in data_file:
       line_split = line.split(',')
       data_instances.append(map(int, line_split))
   
   # 10-fold cross-validation
   ave_success_rate = 0.0
   split_data_instances = np.array_split(data_instances, 10)
   winnow_success = []
   bayes_success = []
   for fold in range(0, 10):
       output_verbose = False
       if fold == 0:
           output_verbose = True
       training_data = np.concatenate((np.delete(split_data_instances, fold, 0)), 0)
       validation_data = split_data_instances[fold]

       # Train the model
       # alpha, theta, number of weights, initial val of weights
       winnow_model = Winnow(2, float(len(data_instances[0][:-1])/2), len(data_instances[0][:-1]), 1)
       naive_bayes = NaiveBayes()
       naive_bayes.build_model(training_data)
       for instance in training_data:
           winnow_model.learn(instance[:-1], instance[-1])
       if output_verbose:
           print("Winnow model = \n%s\n" % winnow_model.print_model())
           print("Bayes model = \n%s\n" % naive_bayes.print_model())

       # Test model
       if output_verbose:
           print("Testing model...")
       trials = len(validation_data)
       for model in ['Winnow', 'Bayes']:
           successes = 0
           failures = 0
           for instance in validation_data:
               if output_verbose:
                   print("Data instance: %s" % instance)
               if model == 'Winnow':
                   prediction = winnow_model.predict(instance[:-1])
               else:
                   prediction = naive_bayes.predict(instance[:-1])
               if output_verbose:
                   print("%s predicts %d" % (model, prediction))
               if prediction != instance[-1]:
                   failures += 1
               else:
                   successes += 1
           if model == 'Winnow':
               winnow_success.append(successes/float(trials))
           else:
               bayes_success.append(successes/float(trials))
   print("Winnow average success rate = %f%% on %s" % (sum(winnow_success)/10 * 100, data_file_name))
   print("Naive bayes average success rate = %f%% on %s\n" % (sum(bayes_success)/10 * 100, data_file_name))
