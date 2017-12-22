import random
import pdb
import numpy as np
from math import sqrt
from math import exp 

class RadialBasisNetwork:
    def __init__(self, learning_rate, spread, num_of_outputs):
        self.input_nodes = []
        self.weights = []
        self.hidden_nodes = []
        self.learning_rate = learning_rate
        self.spread = spread
        self.num_of_outputs = num_of_outputs

    def train(self, training_data):
       # Randomized 10% subset of training data will serve as hidden node centers
       random_centers_for_hidden_nodes = []
       indices = []
       while len(indices) < (training_data.shape[0] * 0.1):
           random_idx = np.random.randint(0, training_data.shape[0])
           if not random_idx in indices:
               indices.append(random_idx)
       random_centers_for_hidden_nodes = training_data[indices]

       # Organize hidden nodes
       for center in random_centers_for_hidden_nodes:
           self.hidden_nodes.append((center[:-1], self.spread))

       # Initial weights for gradient descent, don't forget the bias term
       for _ in range(self.num_of_outputs):
           self.weights.append(
               np.array([np.random.randint(-1, 1) for weight in range(len(self.hidden_nodes) + 1)]))

       # Learn weights to determine hidden node influence on output
       trials = 0
       while trials < 250:
           print(trials)
           # Batch updating of weights, store individual updates in new_weights then divide by number of training_instances and set as new weights
           new_weights = [np.array([0.0 for i in range(len(self.hidden_nodes) + 1)]) for _ in range(self.num_of_outputs)]
           for instance in training_data:
               # Determine Gaussian outputs
               gaussian_outputs = []
               for node in self.hidden_nodes:
                   # radial basis function
                   gaussian_outputs.append( 
                       exp((-1/float(2 * (node[1]**2))) * (sqrt(sum((instance[:-1] - node[0])**2))**2)))
               # Add a one to gaussian_outputs for the bias term
               gaussian_outputs.append(1)

               # Determine error gradient (implies a vector) for each set of output layer weights
               for output_idx in range(self.num_of_outputs):
                   gradient = []
                   for gaussian_output in gaussian_outputs:
                       activation_score = 1 / (1 + exp(-np.dot(
                           self.weights[output_idx], gaussian_outputs)))
                       gradient.append(
                           (activation_score - int((output_idx + 1) == instance[-1])) * 
                           (activation_score * (1 - activation_score)) * gaussian_output)
                   # Calculate weight update
                   # Update weights with w_t+1 = w_t - learning rate * gradient
                   new_weights[output_idx] += (self.weights[output_idx] - (self.learning_rate * np.array(gradient)))
           new_weights = [new_weights[i] / training_data.shape[0] for i in range(self.num_of_outputs)]
           self.weights = new_weights
           trials += 1
       print("Learned weights: %s" % self.weights[output_idx])

    def predict(self, test_set):
        predictions = []
        for instance in test_set:
            gaussian_outputs = []
            for node in self.hidden_nodes:
                # radial basis function
                gaussian_outputs.append( 
                    exp((-1/float(2 * (node[1]**2))) * (sqrt(sum((instance - node[0])**2))**2)))
            # Add a one for the bias term
            gaussian_outputs.append(1)
            outputs = list()
            for weight_idx in range(self.num_of_outputs):
                outputs.append((1 / (1 + exp(-np.dot(self.weights[weight_idx], gaussian_outputs)))))
            predictions.append(outputs.index(np.max(outputs)) + 1)
        return predictions
