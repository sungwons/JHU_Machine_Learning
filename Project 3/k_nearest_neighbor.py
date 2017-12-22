import numpy as np
from math import sqrt

class KNearestNeighbor:
    def __init__(self, k, learner_type):
        self.k = k
        self.learner_type = learner_type

    def train(self, training_data):
        # Stores the training data
        self.training_data = training_data

    def condense_training_data(self):
        print("training data size = %d" % self.training_data.shape[0])
        condensed_training_data = []
        added_to_condensed_set = True
        while added_to_condensed_set:
            added_to_condensed_set = False
            np.random.shuffle(self.training_data)
            for index, instance in enumerate(self.training_data):
                if len(condensed_training_data) == 0:
                    # Add first training instance to start the algorithm
                    condensed_training_data.append(instance)
                    added_to_condensed_set = True
                else:
                    min_distance = float("inf")
                    min_instance = instance
                    for condensed_instance in condensed_training_data:
                        '''Find the instance in the condensed set that is
                           closet to the instance in the original training set'''
                        distance = sqrt(sum((instance[:-1] - condensed_instance[:-1]) ** 2))
                        if distance < min_distance:
                            min_distance = distance
                            min_instance = condensed_instance
                    ''' Compare classification of original instance 
                    and the closet condensed instance'''
                    # if the class labels don't match, add the instance to the condensed set
                    if not instance[-1] == condensed_instance[-1]:
                        if not instance in condensed_instance:
                            print("classes don't match, adding instance")
                            condensed_training_data.append(instance) 
                            added_to_condensed_set = True
        print("Length of new condensed training data set: %d" % len(condensed_training_data))
        self.training_data = np.array(condensed_training_data)

    def predict(self, test_data):
        # This method tries to predict the label values of the input data
        predictions = []
        # Cycle through test data and compute nearest neighbors
        for test_instance in test_data:
            neighbors = []
            for training_instance in self.training_data:
                distance = sqrt(sum((test_instance[:-1] - training_instance[:-1])**2))
                neighbors.append((distance, training_instance))
            neighbors = sorted(neighbors, key=lambda tup: tup[0])[:self.k]
            prediction = 0.0
            # Determine class/regression value by averaging neighbors
            for neighbor in neighbors:
                prediction += neighbor[1][-1]
            prediction = prediction / len(neighbors)
            # Round output if we are doing classification
            if self.learner_type == "CLASSIFICATION":
                prediction = int(round(prediction))
            predictions.append(prediction)
        return predictions
