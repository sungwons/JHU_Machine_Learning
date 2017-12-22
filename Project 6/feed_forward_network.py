import numpy as np
import pdb
from math import exp

class FeedForwardNetwork:
    def __init__(self, learning_rate, num_of_attrs, 
                 num_of_hidden_layers, num_of_hidden_nodes_per_layer, 
                 num_of_outputs):

        # This method initializes the instances variables.
        self.learning_rate = learning_rate
        self.num_of_attrs = num_of_attrs
        self.num_of_outputs = num_of_outputs
        self.num_of_hidden_nodes_per_layer = num_of_hidden_nodes_per_layer
        self.network = list()

        # Initialize the network structure and the weights for each hidden node
        for layer_idx in range(num_of_hidden_layers):
            layer = list()
            for node_idx in range(num_of_hidden_nodes_per_layer[layer_idx]):
                if layer_idx == 0:
                    layer.append({'weights': np.random.uniform(0, 1, num_of_attrs + 1)})
                else:
                    layer.append({'weights': np.random.uniform(0, 1, num_of_hidden_nodes_per_layer[layer_idx - 1] + 1)})
            self.network.append(layer)

        # Initialize the output node weights
        output_layer = list()
        if num_of_hidden_layers > 0:
           for layer_idx in range(self.num_of_outputs):
               output_layer.append({'weights': np.random.uniform(
                       0, 1, num_of_hidden_nodes_per_layer[-1] + 1)})
        else:
           for layer_idx in range(self.num_of_outputs):
               output_layer.append({'weights': np.random.uniform(
                       0, 1, num_of_attrs + 1)})
        self.network.append(output_layer)

    def determine_node_outputs(self, data_instance):
        '''This method determines the each node's output 
           by feeding the data instance to the network and 
           propagating the signal through to the output nodes.
           The logistic function is used as the activation 
           function.
        '''
        current_layer_input = data_instance
        for layer in self.network:
            next_layers_input = list()
            for node in layer:
                # Calculate the node's output with the dot product of the weights and then add on the bias.

                if len(node['weights'][:-1]) != len(current_layer_input):
                    pdb.set_trace()
                dot_prod = np.dot(node['weights'][:-1], current_layer_input) + node['weights'][-1]
                node['output'] = 1.0 / (1.0 + exp(-dot_prod))
                next_layers_input.append(node['output'])
            current_layer_input = next_layers_input

    def backprop_error(self, targeted_network_value):

        '''This method calculates the error each node is
           responsible for by backpropagating the error from the
           output nodes to the input layer of the network.
        '''
        output_layer_index = len(self.network) - 1
        for layer_idx in reversed(range(len(self.network))):
            layer = self.network[layer_idx]
            # if this is the output layer, calculate the error according to step 1
            if layer_idx == output_layer_index:
                # iterate over the nodes in the output layer and calculate the error
                for output_node, expected_node_output in zip(layer, targeted_network_value):
                    output = output_node['output']
                    output_node['error'] = \
                        output * (1.0 - output) * (expected_node_output - output)
            else:
                # else calculate error according to step 2 
                for hidden_layer_node_idx, hidden_layer_node in enumerate(layer):
                    error_sum = 0.0
                    # first backprop error from the downstream nodes
                    for downstream_node in self.network[layer_idx + 1]:
                        error_sum += \
                            downstream_node['error'] * \
                            downstream_node['weights'][hidden_layer_node_idx]
                    output = hidden_layer_node['output']
                    # calculate this node's error
                    hidden_layer_node['error'] = \
                        output * (1.0 - output) * error_sum

    def train(self, training_set):
        '''This method trains the weights of the network using
           backpropagation and the logistic function as its
           activation function.
        '''
        trials = 0
        while trials < 250:
            for instance in training_set:
                # Calculate the actual outputs for each node
                self.determine_node_outputs(instance[:-1])
                # Backprop the error through the network, this function
                # takes a hot-encoding of the classes
                self.backprop_error( \
                    [1 if (instance[-1] - 1) == idx else 0 for idx in range(self.num_of_outputs)])

                for layer_idx, layer in enumerate(self.network):
                    for node in layer:
                        if layer_idx == 0:
                            # first hidden layer
                            for weight_idx in range(len(node['weights']) - 1):
                                node['weights'][weight_idx] = \
                                    node['weights'][weight_idx] + \
                                    self.learning_rate * node['error'] * instance[weight_idx]
                        else:
                            # hidden layer or output layer
                            # for weight_idx in xrange(self.num_of_attrs):
                            for weight_idx in range(len(node['weights']) - 1):
                                node['weights'][weight_idx] = \
                                    node['weights'][weight_idx] + \
                                    self.learning_rate * node['error'] * \
                                    self.network[layer_idx - 1][weight_idx]['output']
                        # Update the bias
                        node['weights'][-1] = \
                            node['weights'][-1] + self.learning_rate * node['error']
            trials += 1

    def predict(self, test_set):
        predictions = list()
        for instance in test_set:
            # Propagate the instance through the network
            self.determine_node_outputs(instance)
            # Retrieve the output layer's outputs and choose the highest
            outputs = [self.network[-1][i]['output'] for i in range(self.num_of_outputs)]
            predictions.append(outputs.index(np.max(outputs)) + 1)
        return predictions

    def __str__(self):
        '''This method prints the network.'''
        return_val = "Hidden nodes: "
        for layer in self.network[:-1]:
            return_val += str(layer)
        return_val += '\nOutput layer: '
        return_val += str(self.network[-1])
        return return_val
