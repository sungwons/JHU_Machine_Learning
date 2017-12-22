import numpy as np
import pdb
import random

class Adaline:
    def __init__(self, bias, learning_rate):
        self.bias = bias
        self.learning_rate = learning_rate

    def train(self, training_set):
        # Initialize the weights
        self.weights = np.array([random.uniform(-0.01, 0.01) for i in range(training_set.shape[1] - 1)])

        # While the weights haven't converged...
        converged = False
        iterations = 0
        while not converged and iterations < 5000:
            old_weights = self.weights
            weight_update = np.array([0.0 for i in range(training_set.shape[1] - 1)])
            for instance in training_set:
                # Calculate their dot product with the weights
                output = np.dot(instance[:-1], self.weights) - self.bias
                # Calculate update
                weight_update += self.learning_rate * (instance[-1] - output) * instance[:-1]
            # Average updates and apply to weights (batch updating)
            self.weights = self.weights + (weight_update / training_set.shape[0])
                 
            if sum(abs(old_weights - self.weights)) < 0.0001:
                converged = True
            iterations += 1
        print("Learned adaline weights: %s" % str(self.weights))

    def predict(self, test_set):
        #This method predicts the class for each example in the test set
        predictions = []
        # Cycle through instances
        for instance in test_set:
            # Calculate their dot product with the weights and subtract
            # off bias
            output = np.dot(instance, self.weights) - self.bias
            # This if-else serves as a sigmoid activiation function,
            # where sgn(x) = 1 when output > 0 and 0 otherwise
            if output > 0:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions
