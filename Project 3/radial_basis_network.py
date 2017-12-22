import numpy as np
from math import sqrt
from math import exp 
from k_means import KMeans

class RadialBasisNetwork:
    def __init__(self, learning_rate, learner_type):
        self.input_nodes = []
        self.weights = []
        self.hidden_nodes = []
        self.learner_type = learner_type
        self.learning_rate = learning_rate

    def train(self, training_data, do_clustering = False, num_of_means = -1):

       if do_clustering:
           # Run k-means on the training set, the centroids become the middle of the hidden nodes
           kmeans_model = KMeans(num_of_means)
           kmeans_model.cluster(training_data)
           centroids = kmeans_model.get_centroids()
           # Determine the spread for each hidden node, spread
           for cent_1_idx, cent_1_val in enumerate(centroids):
               total_distance = 0.0
               for cent_2_idx, cent_2_val in enumerate(centroids):
                   if cent_1_idx != cent_2_idx:
                       # Calculate distance between two centroids
                       total_distance += sqrt(sum((np.array(cent_1_val[:-1]) - np.array(cent_2_val[:-1])) ** 2))
               # Spread defined above, centroid = hidden node center
               self.hidden_nodes.append(
                   (cent_1_val, 2 * (total_distance / (training_data.shape[0] - 1))))
       else: 
           # Randomized 10% subset of training data will serve as hidden node centers
           random_centers_for_hidden_nodes = []
           indices = []
           while len(indices) < (training_data.shape[0] * 0.1):
               random_idx = np.random.randint(0, training_data.shape[0])
               if not random_idx in indices:
                   indices.append(random_idx)
           random_centers_for_hidden_nodes = training_data[indices]

           # Determine the spread for each hidden node, spread
           for inst_1_idx, inst_1_val in enumerate(random_centers_for_hidden_nodes):
               total_distance = 0.0
               for inst_2_idx, inst_2_val in enumerate(random_centers_for_hidden_nodes):
                   if inst_1_idx != inst_2_idx:
                       # Calculate distance between two instances
                       total_distance += sqrt(sum((inst_1_val[:-1] - inst_2_val[:-1]) ** 2))
               # Spread defined above, instance = hidden node center
               self.hidden_nodes.append(
                   (inst_1_val[:-1], 2 * (total_distance / (len(indices) - 1))))
           print("Chose random training instances to serve as hidden node centers")

       # Initial weights for gradient descent
       self.weights = \
           np.array([np.random.randint(-100, 100) for weight in range(len(self.hidden_nodes))])
       print("Initialized weight vector")

       # Learn weights to determine hidden node influence on output
       done = False
       while not done:
           print("Started gradient descent")
           # Batch updating of weights, store individual updates in new_weights
           new_weights = np.array([0.0 for i in range(len(self.weights))])
           for instance in training_data:
               # Determine Gaussian outputs
               gaussian_outputs = []
               for node in self.hidden_nodes:
                   # radial basis function
                   gaussian_outputs.append( 
                       exp((-1/float(2 * (node[1]**2))) * (sqrt(sum((instance[:-1] - node[0])**2)))))

               # Determine error gradient (implies a vector)
               gradient = []
               for gaussian_output in gaussian_outputs:
                   if self.learner_type == "REGRESSION":
                       gradient.append(
                           2 * (np.dot(self.weights, gaussian_outputs) - instance[-1]) * 
                           gaussian_output)
                   else:
                       activation_score = 1 / (1 + exp(np.dot(self.weights, gaussian_outputs)))
                       gradient.append(
                           (activation_score - instance[-1]) * 
                           (activation_score * (1 - activation_score)) * gaussian_output)

               # Calculate weight update
               new_weights += (self.weights - (self.learning_rate * np.array(gradient)))
           new_weights = (new_weights / training_data.shape[0])
           if (abs(sum(self.weights - new_weights))) < 0.01:
               done = True
           if not done:
               print("Weights were updated by %f last iteration" % abs(sum(self.weights - new_weights)))
           self.weights = new_weights
       print("Found weights: %s" % str(self.weights))

    def predict(self, test_set):
        predictions = []
        for instance in test_set:
            gaussian_outputs = []
            for node in self.hidden_nodes:
                # radial basis function
                gaussian_outputs.append(
                    exp((-1/float(2 * (node[1]**2))) * (sqrt(sum((instance[:-1] - node[0])**2)))))
            if self.learner_type == "REGRESSION":
                predictions.append(np.dot(self.weights, gaussian_outputs))
            else:
                activation_value = (1 / (1 + exp(np.dot(self.weights, gaussian_outputs))))
                if activation_value < 0.5:
                    predictions.append(0)
                else:
                    predictions.append(1)
        return predictions
