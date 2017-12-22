from __future__ import division
from math import log
from decision_tree_node import DecisionTreeNode
import pdb
from copy import deepcopy
import numpy as np

'''This class implements a classification decision tree using gain ratio.'''
class ClassificationDecisionTree:
    def __init__(self, attr_info, class_labels):
        # attr_info is a list of tuples where tuple[0] is the
        # attribute type ("DISCRETE", "CONTINUOUS") and tuple[1]
        # is a list of discrete values the attribute takes or
        # ranges to consider
        """

        :rtype: object
        """
        self.attr_info = attr_info
        self.class_labels = class_labels
        self.best_split_value_cache = {}

    def calculate_information(self, split):
        '''
            This function calculates the information of a problem
            given a subset of instances split on the values (j) of a particular
            attribute (f_i).
          I(c_1,...,c_k) = -sum from l = 1 to k (c_l / (c_1+...+c_k)) log (c_l / (c_1+...+c_k)))
          where 
              c_l is the is the number of examples w/ label c_l and f_i = j
              c_1+...+c_k is the number of examples w/ f_i = j
        '''
        total = 0.0
        # Iterate over class labels
        for label in self.class_labels:
            # Calculate (c_1 / (c_1+...+c_k)) log (c_1 / (c_1+...+c_k))
            class_proportion = 0
            if split.shape[0] != 0:
                class_proportion = \
                    (len(np.where(split[:,-1] == label)[0])) / float(split.shape[0])
            if class_proportion != 0:
                total += class_proportion * log(class_proportion,2)
        return -total

    def calculate_entropy(self, instances, attr_idx, attr_type="DISCRETE", split_value=-1.0):
        '''
            This function calculates the entropy of a group of instances
            given a specific attribute.

            entropy(attr_idx) = 
                sum over domain of attribute at attr_idx 
                    (examples with attr_idx = attr_val / # of examples) * information(split)
        '''
        total = 0.0
        if attr_type == "DISCRETE":
            # For each value of attribute at attr_idx, calculate the entropy
            for attr_val in self.attr_info[attr_idx][1]:
                # Find subset of instances where value(attr_idx) = attr_val
                split = instances[np.where(instances[:, attr_idx] == attr_val)]
                # entropy(attr_val) = 
                #     (a / b) * information(split)
                # 
                #     where a = # of examples w/ f_i = attr_val
                #           b = # of examples in instances
                total += (split.shape[0] / float(instances.shape[0])) * \
                         self.calculate_information(split)
        else:
            for operation in ['<=', '>']:
                # Create subsets using 'less than or equal' to and 'greater than'
                if operation == '<=':
                    split = instances[np.where(instances[:, attr_idx] <= split_value)]
                else:
                    split = instances[np.where(instances[:, attr_idx] > split_value)]
                total += (split.shape[0] / float(instances.shape[0])) * \
                        self.calculate_information(split)
        return total

    def calculate_gain(self, split, attr_idx, split_point=float("inf")):
        '''
            This function calculates gain.
                gain = information(split) - entropy(split, attr)
        '''
        if self.attr_info[attr_idx][0] == "DISCRETE":
            return self.calculate_information(split) - \
                self.calculate_entropy(split, attr_idx)
        else:
            # If the split value of the numeric attribute is specified
            # use it. Otherwise, figure it out.
            if split_point == float("inf"):
                return self.calculate_information(split) - \
                    self.calculate_entropy(split, attr_idx, "CONTINUOUS", 
                        self.determine_best_split_value_for_cont_variable(split, attr_idx))
            else:
                return self.calculate_information(split) - \
                    self.calculate_entropy(split, attr_idx, "CONTINUOUS", split_point)

    def determine_best_split_value_for_cont_variable(self, split, attr_idx):
        '''
            I'm using the method outlined in Tom Michell's Machine Learning textbook
            to determine what value to split a numeric attribute.
            Specifically: 
                1. Sort the instances on attr_idx from low to high.
                2. Create a list of candidate split points from the 
                   mid-points of the attribute values between 
                   instances where the class label changes.
                3. Calculate the gain for each split point and return 
                   the one with the highest gain.
        '''
        # To speed things up, let's keep a cache of previously calculated values.
        # The cache is map of tuple(hash(instances), attribute index) to split value.

        # Do some stuff to make numpy happy
        split_copy = np.copy(split)
        split_copy.flags.writeable = False
        hash_val = hash(split_copy.data)

        # If we've already calculated the best split for this combo
        # of instances and attribute index, just return it.
        if (hash_val, attr_idx) in self.best_split_value_cache.keys():
            return self.best_split_value_cache[(hash_val, attr_idx)]

        # We only need the column attr_idx and the class label in order
        # to generate a candidate list of values.
        subset = np.column_stack((split[:, attr_idx], split[:, -1]))

        # Sort on attr_idx from low to high
        subset = subset[subset[:,0].argsort()]
        previous_instance = subset[0]
        candidate_split_points = []
        for instance in subset:
            # Calculate the midpoint's on attr_idx where the class label changes
            if previous_instance[-1] != instance[-1]:
                candidate_split_points.append((float(previous_instance[0]) + instance[0]) / 2)
            previous_instance = instance

        # Cycle through candidate split points and return the one with the high gain
        max_gain = float("-inf")
        max_gain_split_point = -1.0
        for split_point in candidate_split_points:
            gain = self.calculate_gain(split, attr_idx, split_point)
            if gain > max_gain:
                max_gain = gain
                max_gain_split_point = split_point
        self.best_split_value_cache[(hash_val, attr_idx)] = max_gain_split_point
        return max_gain_split_point

    def calculate_intrinsic_value(self, split, attr_idx):
        '''This method calculates the intrinsic value of a group of instances
           given a specific attribute.'''
        total = 0.0
        # Is this attribute numeric or discrete?
        if self.attr_info[attr_idx][0] == "DISCRETE":
            # Cycle through each valid value of this attribute,
            # keeping a running total for the intrinsic value calculation
            for attr_val in self.attr_info[attr_idx][1]:
                # Intrinsic value calculation
                count = len(np.where(split[:, attr_idx] == attr_val)) / float(split.shape[0])
                if count != 0:
                    total += count * log(count,2)
        else:
            # Determine the best split value
            attr_val = self.determine_best_split_value_for_cont_variable(split, attr_idx)
            for operation in ['<=', '>']:
                # Intrinsic value calculation
                if operation == '<=':
                    count = \
                        len(np.where(split[:, attr_idx] <= attr_val)) / float(split.shape[0])
                else:
                    count = \
                        len(np.where(split[:, attr_idx] > attr_val)) / float(split.shape[0])
                if count != 0:
                    total += count * log(count,2)
        return -total

    def determine_split_attribute(self, instances, attributes):
        '''This method determines the best split attribute
           given a group of instances and a list of attributes.'''
        max_gain_ratio = float("-inf")
        best_attr_idx = -1
        # For every attribute
        for attr_idx in attributes:
               # Split instances based on value of attribute
               # and calculate gain_ratio
               gain_ratio = 0
               iv = self.calculate_intrinsic_value(instances, attr_idx)
               if (iv != 0):
                   gain_ratio = \
                       self.calculate_gain(instances, attr_idx) / \
                       self.calculate_intrinsic_value(instances, attr_idx)
               # if cal_gain_ratio < max_gain_ratio, reset min and index
               if gain_ratio > max_gain_ratio:
                   max_gain_ratio = gain_ratio
                   best_attr_idx = attr_idx
        # return best attribute
        return best_attr_idx

    def generate_tree(self, instances, attributes):
        '''This method is the main ID3 loop. It starts by making
           simple check on the group of instances it is given
           and makes leaf nodes if necessary. If not, it proceeds
           to the main ID3 learning loop: in a breadth-first manner,
           determine the attribute with the highest gain ratio from
           the list of available attributes and group of instances and
           split on that attribute.'''
        # Intialize the root node
        self.root_node = DecisionTreeNode()
        self.root_node.instances = instances

        # Initialize the job queue, this queue is what makes the 
        # algorithm run in a breadth-first manner.
        job_queue = []
        job_queue.append((self.root_node, instances, attributes))
        while len(job_queue) > 0:
            # Grab the next node off the queue
            node, inst, attrs = job_queue.pop(0)
            
            # Are we out of attributes to split on or are all the instances
            # in this group of the same class? If so, just create a leaf node
            # and be done with it.
            if len(attrs) == 0 or np.unique(inst[:,-1]).shape[0] == 1:
                node.label = np.argmax(np.bincount(inst[:,-1].astype(int)))
                node.instances = inst
                continue
            else: 
                # Determine the attribute with the best gain ratio,
                # and set the node's split information accordingly
                split_attr = self.determine_split_attribute(inst, attrs) 
                node.split_attr = split_attr
                node.split_attr_type = self.attr_info[split_attr][0]
                if node.split_attr_type == "DISCRETE":
                    node.attr_values = np.unique(inst[:,split_attr])
                else:
                    node.attr_values = \
                        self.determine_best_split_value_for_cont_variable(inst, split_attr)

                # Remove the chosen attribute from the list of available attributes
                attributes.remove(split_attr) 

                # Create the necessary branches from this node based on the 
                # values the split attribute can take
                if self.attr_info[split_attr][0] == "DISCRETE":
                    # for each value that split_attr can take...
                    for attr_val in self.attr_info[split_attr][1]: 
                        # Determine instances with that value and 
                        # create new nodes for them, add them to the queue
                        subset = inst[np.where(inst[:,split_attr] == attr_val)]
                        if subset.shape[0] > 0:
                            child_node = DecisionTreeNode()
                            child_node.instances = subset
                            job_queue.append((child_node, subset, attrs))
                            node.add_branch(child_node)
                else:
                    # For a numeric attribute, we use a binary split given a split value
                    subset_leq = inst[np.where(inst[:,split_attr] <= node.attr_values)]
                    subset_greater = inst[np.where(inst[:,split_attr] > node.attr_values)]
                    if subset_leq.shape[0] > 0 and subset_greater.shape[0] > 0: 
                        # Determine instances with that value and 
                        # create new nodes for them, add them to the queue
                        child_node = DecisionTreeNode() 
                        child_node.instances = subset_leq
                        job_queue.append((child_node, subset_leq, attrs)) 
                        node.add_branch(child_node)
                        # Determine instances with that value and 
                        # create new nodes for them, add them to the queue
                        child_node2 = DecisionTreeNode() 
                        child_node2.instances = subset_greater
                        job_queue.append((child_node2, subset_greater, attrs)) 
                        node.add_branch(child_node2)
                    else:
                        # Leaf node!
                        node.branches = []
                        node.label = np.argmax(np.bincount(inst[:,-1].astype(int)))
                        node.instances = inst
        return self.root_node 

    def listifyTree(self, tree):
        '''This method turns a tree (given as a root node) into 
           a list consisting of all the nodes in the tree.'''
        listified_tree = [tree]
        for branch in tree.branches:
            listified_tree += self.listifyTree(branch)
        return listified_tree

    def prune(self, validation_set):
        '''This method performs reduced-error pruning on the 
            learned decision tree given a validation set.'''
        test_tree = ClassificationDecisionTree(self.attr_info, self.class_labels)
        # While pruning is still helping...
        while True:
            # Make a copy of the tree
            test_tree.root_node = deepcopy(self.root_node)
            # Iterate through all the nodes, prune their subtrees, grade on the validation set
            best_tree = None
            best_performance = float("-inf")
            # Create a single list of all the nodes in the tree so
            # we can rip through them
            nodes_to_check = self.listifyTree(test_tree.root_node)
            # Rip through list of nodes, nullifying each, testing its performance
            # and recording the best performance
            for node in nodes_to_check:
                if len(node.branches) > 0:
                    node_copy = deepcopy(node)
                    # Nullify node
                    node.branches = []
                    node.label = np.argmax(np.bincount(node.instances[:,-1].astype(int)))

                    # Calculate performance on validation set
                    performance = \
                        sum(test_tree.predict(validation_set[:,:-1]) == validation_set[:, -1]) / \
                        float(validation_set.shape[0])
                    if performance > best_performance:
                        best_performance = performance
                        best_tree = test_tree.root_node
                    # Un-nullify node
                    node = node_copy
            # Compare best pruned performance with the 
            # full tree's performance on the validation set
            predictions = self.predict(validation_set[:,:-1])
            unpruned_performance = \
                sum(predictions == validation_set[:,-1]) / float(validation_set.shape[0])
            if unpruned_performance <= best_performance:
                print("Pruned tree improved performance by %f" % \
                abs(unpruned_performance - best_performance))
                # Apply the prune
                self.root_node = best_tree
            else:
                break
        print("Pruned tree: ", self.print_tree(self.root_node))

    def train(self, training_data, attributes):
        '''This method trains the decision tree by calling a helper function.'''
        self.root_node = self.generate_tree(training_data, attributes)
        print("Learned tree: ", self.print_tree(self.root_node))

    def predict(self, test_data):
        '''This method predicts the class labels of the given instances.
           Note: class labels should be removed of the passed in instances.'''
        predictions = []
        # Cycle through the test instances...
        for instance in test_data:
            # Traverse the tree, starting at the root node based on the
            # values of this test instance for the attributes dicated by the nodes
            # of the learned tree
            node = self.root_node
            # only leaf nodes have no branches, so while we aren't at a 
            # leaf node...
            while len(node.branches) != 0:
                # Choose the correct branch to traverse
                if node.split_attr_type == "DISCRETE":
                    node = \
                        node.branches[np.where(node.attr_values == instance[node.split_attr])[0]]
                else:
                    if instance[node.split_attr] <= node.attr_values:
                        node = node.branches[0]
                    else:
                        node = node.branches[1]
            # Finally we're at a leaf node! Return the class label stored here
            predictions.append(node.label)
        return predictions

    def print_tree(self, root, indent=0):
        '''This method prints out the tree given a root note.'''
        # Print out root node
        result = '\n' + (' ' * indent) + "node(" + str(root)
        node = root
        # If the root has branches, print those
        if len(node.branches) > 0:
            result += '\n' + (' ' * indent) + 'branches('
        idx = 0
        while idx < len(node.branches):
            result += self.print_tree(node.branches[idx], indent+4)
            idx += 1
        # Close up branches
        if len(node.branches) > 0:
            result += ')'
        # Close up node
        result += ")"
        return result
