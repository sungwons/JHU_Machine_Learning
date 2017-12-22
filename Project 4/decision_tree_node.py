from __future__ import division
import pdb
import numpy as np

class DecisionTreeNode:
    '''This class encapsulates a simple tree node, which contains information like
       the split attribute used, the valid values for that split attribute, the branches
       connected to this node, and a class label (if this node is a leaf node).'''
    def __init__(self):
        '''This method initializes the node.'''
        self.split_attr = -1
        self.split_attr_type = "DISCRETE"
        self.attr_values = []
        self.label = -1
        self.branches = []
        self.instances = None

    def add_branch(self, node):
        '''This method adds a branch to this node.'''
        self.branches.append(node)

    def __str__(self):
        '''This method pretty prints the node.
           Note: nodes are printed differently depending on if they 
           are an interior or leaf node.'''
        if len(self.branches) == 0:
            node_str = "class label: %d" % self.label
        else:
            node_str = 'split attribute: %d, split attribute type: %s, split values: %s, split branches length: %d, split number of instances: %d' % \
                (self.split_attr, self.split_attr_type,
                 str(self.attr_values), len(self.branches), len(self.instances))
        return node_str

    def __repr__(self):
        '''This method pretty prints the node during debugging.'''
        return self.__str__()
