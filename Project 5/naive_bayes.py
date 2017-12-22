import numpy as np

class NaiveBayes:
     # This class encapsulates the naive Bayes algorithm.

    def __init__(self):
        self.class_priors = {0: 0, 1: 0}
        # Represents table of conditional probabilities,
        self.feature_given_class_prob = {0: {0: {}, 1: {}}, 1: {0: {}, 1: {}}}

    def build_model(self, data_instances):
       # Split on class
       negative_instances = []
       positive_instances = []
       for instance in data_instances:
           if instance[-1] == 0:
               negative_instances.append(instance[:-1])
           elif instance[-1] == 1:
               positive_instances.append(instance[:-1])

       # Calculate class priors 
       self.class_priors[0] = float(len(negative_instances)) / len(data_instances)
       self.class_priors[1] = float(len(positive_instances)) / len(data_instances)
       negative_instances = np.array(negative_instances)
       positive_instances = np.array(positive_instances)

       # Calculate feature given class
       for class_value in range(2):
           for feature_value in range(2):
               for feature_idx in range(len(instance[:-1])):
                   if class_value == 0:
                       if len(negative_instances) > 0:
                           self.feature_given_class_prob[class_value][feature_value][feature_idx] = \
                               float((negative_instances[:,feature_idx] == feature_value).sum()) / len(negative_instances)
                       else:
                           self.feature_given_class_prob[class_value][feature_value][feature_idx] = 0.0
                   if class_value == 1:
                       if len(positive_instances) > 0:
                           self.feature_given_class_prob[class_value][feature_value][feature_idx] = \
                               float((positive_instances[:,feature_idx] == feature_value).sum()) / len(positive_instances)
                       else:
                           self.feature_given_class_prob[class_value][feature_value][feature_idx] = 0.0

    def predict(self, data_instance):
        argmax_class_probability = -1.0
        argmax_class = -1
        for class_value in range(2):
            probability_of_belonging_to_class_value = self.class_priors[class_value]
            for attr_idx, instance_attr in enumerate(data_instance): 
                if self.feature_given_class_prob[class_value][instance_attr][attr_idx] == 0:
                    probability_of_belonging_to_class_value *= 0.00000000000000000000001
                else:
                    probability_of_belonging_to_class_value *= \
                        self.feature_given_class_prob[class_value][instance_attr][attr_idx]
            if probability_of_belonging_to_class_value > argmax_class_probability:
                argmax_class_probability = probability_of_belonging_to_class_value
                argmax_class = class_value
        return argmax_class

    def print_model(self):
        model = "P(class value = {}) = {}\n".format(0, self.class_priors[0])
        model += "P(class value = {}) = {}\n".format(1, self.class_priors[1])
        for class_key, inst_val_map in self.feature_given_class_prob.iteritems():
            for instance_val_key, value in inst_val_map.iteritems():
                for attr_key, probability in value.iteritems():
                    model += "P(f_{} = {} | Class value = {}) = {}\n".format(  \
                        attr_key, instance_val_key, class_key, probability)
        return model
