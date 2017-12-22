from __future__ import division
import numpy as np
from math import exp
import random

class LogisticRegression:
    def __init__(self, learning_rate, class_labels):
        # This method is simple initializer for the class.
        self.class_labels = class_labels
        self.class_weights = {}
        self.learning_rate = learning_rate

    def initialize_weights(self, num_of_attr):
        # This method initializes the weights for each class to a random float between -0.01 and 0.01.
        for class_label in self.class_labels:
            self.class_weights[class_label] = [random.uniform(-0.01, 0.01) for i in range(num_of_attr)]

    def calculateProbabilityClassGivenInstance(self, instances, class_label):
        # This method calculates P(C=c_k | instance) for a group of instances and a class label.

        # for each instance, calculate P(C=c_k | instance)
        probabilities = []
        for instance in instances:
            linear_sum = np.dot(self.class_weights[class_label], instance)
            probabilities.append(exp(linear_sum) / (1 + exp(linear_sum)))
        return np.transpose(np.array(probabilities))

    def train(self, training_set, num_of_attr):
        # This method learns the weights for each class.
        # Initialize weights
        self.initialize_weights(num_of_attr)

        # For each class, calculate its weight vector
        for class_label in self.class_labels:
            print("Learning weights for class %d" % class_label)
            # While the weights haven't converged, calculate the change in the weight vector by iterating through the training data
            converged = False
            iterations = 0
            while not converged and iterations < 10000:
                # Batch updating of weights: store individual updates in new_weights then divide by number of training_instances and set as new weights
                new_weights = \
                    np.array([0.0 for i in range(training_set.shape[1] - 1)])
                '''Calculate new weights as follows:
                    w_{ji+1} = w_{ji} + 
                        learning_rate * 
                        sum over examples (x_{i}^{k} * 
                                           Kronecker delta - P(C^k = c_j | x^k, w))'''
                # Calculate P(c^k = c_j | x^k, weight vector)
                second_term = \
                    self.calculateProbabilityClassGivenInstance(training_set[:, :-1], class_label)
                for attr in range(training_set.shape[1] - 1):
                    # Calculate x_{i}^{k} * Kronecker delta
                    first_term = \
                        training_set[:, attr] * (training_set[:, -1]==class_label).astype(int)

                    weight_delta = np.sum(first_term - second_term) #/ training_set.shape[0]
                    # new_weight = old_weight * learning_rate * weight_delta
                    new_weights[attr] += \
                        self.class_weights[class_label][attr] + self.learning_rate * weight_delta

                weight_update_size = abs(sum(self.class_weights[class_label] - new_weights))
                if weight_update_size < 0.001:
                    converged = True
                    print("Learned weights of %s for class %d" % (new_weights, class_label))
                if not converged:
                    print("Weights for class %d were updated by %f" % \
                        (class_label, weight_update_size))
                iterations += 1
                self.class_weights[class_label] = new_weights

    def predict(self, test_set):
        # This method predicts the output of every instance in the test set.
        predictions = []
        # Run through each instance...
        for instance in test_set:
            # Keep a list of probabilities (one list element for each class)
            probabilities = []

            for class_label in self.class_labels[:-1]:
                # output = w_0 + sum_{i=1}^{d} (w_i * x_i)
                output = np.dot(self.class_weights[class_label], instance)
                probabilities.append(exp(output) / (1 + exp(output)))
            # Add probability of last class
            probabilities.append(1 - sum(probabilities))

            # The predicted class label is equal to the class with the highest probability of class_label | instance
            predictions.append(probabilities.index(max(probabilities)))

        return predictions
