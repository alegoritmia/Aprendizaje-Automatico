# https://towardsdatascience.com/id3-decision-tree-classifier-from-scratch-in-python-b38ef145fd90
# ID3 (Iterative Dichotomiser 3) Algorithm implementation from scratch

import math
from collections import deque

class Node:
    """Contains the information of the node and another nodes of the Decision Tree."""
    """We connect an attribute with another attribute or class. In the middle, we will have attribute values as nodes"""

    def __init__(self):
        """
        value: Feature to make the split and branches.
        next: Next node This can be an attribute or a class.
        childs: Branches coming off the decision nodes, branches are values of attributes
        """
        self.value = None
        self.next = None
        self.childs = None


class DecisionTreeClassifier:
    """Decision Tree Classifier using ID3 algorithm."""

    def __init__(self, X, attribute_names, y):
        self.X = X
        self.y = y
        self.attribute_names = attribute_names
        self.uniqueClassLabels = list(set(y)) 
        self.uniqueClassLabelsCount = [list(y).count(x) for x in self.uniqueClassLabels]
        self.node = None
        self.entropy = self._get_entropy([x for x in range(len(self.y))])  # calculates the initial entropy

    def _get_entropy(self, x_ids):
        """ Calculates the entropy.
        Parameters
        __________
        :param x_ids: list, List containing the instances ID's
        __________
        :return: entropy: float, Entropy.
        """
        # sorted y by instance id
        y = [self.y[i] for i in x_ids]
        # count number of instances of each category
        label_count = [y.count(x) for x in self.uniqueClassLabels]
        # calculate the entropy for each category and sum them
        entropy = sum([-count / len(x_ids) * math.log(count / len(x_ids), 2) if count else 0 for count in label_count])
        return entropy

    def _get_information_gain(self, x_ids, attribute_idx):
        """Calculates the information gain for a given feature based on its entropy and the total entropy of the system.
        Parameters
        __________
        :param x_ids: list, List containing the instances ID's
        :param attribute_idx: int, feature ID
        __________
        :return: info_gain: float, the information gain for a given feature.
        """
        # calculate total entropy
        info_gain = self._get_entropy(x_ids)
        # store in a list all the values of the chosen feature
        x_features = [self.X[x][attribute_idx] for x in x_ids]
        # get unique values
        feature_vals = list(set(x_features))
        # get frequency of each value
        feature_vals_count = [x_features.count(x) for x in feature_vals]
        # get the feature values ids
        feature_vals_id = [
            [x_ids[i]
            for i, x in enumerate(x_features)
            if x == j]
            for j in feature_vals
        ]

        # compute the information gain with the chosen feature
        info_gain = info_gain - sum([val_counts / len(x_ids) * self._get_entropy(val_ids)
                                     for val_counts, val_ids in zip(feature_vals_count, feature_vals_id)])

        return info_gain

    def _get_feature_max_information_gain(self, x_ids, attribute_idxs):
        """Finds the attribute that maximizes the information gain.
        Parameters
        __________
        :param x_ids: list, List containing the samples ID's
        :param attribute_idxs: list, List containing the feature ID's
        __________
        :returns: string and int, name and attribute id of the attribute that maximizes the information gain
        """
        # get the entropy for each feature
        information_gain = [self._get_information_gain(x_ids, attribute_idx) for attribute_idx in attribute_idxs]
        # find the feature that maximises the information gain
        max_id = attribute_idxs[information_gain.index(max(information_gain))]

        return self.attribute_names[max_id], max_id

    def id3(self):
        """Initializes ID3 algorithm to build a Decision Tree Classifier.
        :return: None
        """
        x_ids = [x for x in range(len(self.X))] # m
        attribute_idxs = [x for x in range(len(self.attribute_names))] # n
        self.node = self._id3_recv(x_ids, attribute_idxs, self.node) # first time self.node is None
        print('')

    def _id3_recv(self, x_ids, attribute_idxs, node):
        """ID3 algorithm. It is called recursively until some criteria is met.
        Parameters
        __________
        :param x_ids: list, list containing the samples ID's
        :param attribute_idxs: list, List containing the feature ID's
        :param node: object, An instance of the class Nodes
        __________
        :returns: An instance of the class Node containing all the information of the nodes in the Decision Tree
        """
        if not node:
            node = Node()  # initialize nodes

        # sorted y by instance id : Obtaining the classes for each instance
        y_in_features = [self.y[x] for x in x_ids]

        # if all the examples have the same class (pure node), return node
        if len(set(y_in_features)) == 1:
            node.value = self.y[x_ids[0]]
            return node
        # if there are not more feature to compute, return node with the most probable class
        if len(attribute_idxs) == 0:
            node.value = max(set(y_in_features), key=y_in_features.count)  # compute mode
            return node
        # else...
        # choose the feature that maximizes the information gain
        best_feature_name, best_attribute_idx = self._get_feature_max_information_gain(x_ids, attribute_idxs)
        node.value = best_feature_name
        print('Setting node value as {}'.format(node.value))
        node.childs = []
        # value of the chosen feature for each instance
        feature_values = list(set([self.X[x][best_attribute_idx] for x in x_ids]))
        # loop through all the values. The objective is to find the next node or class
        for value in feature_values:
            # For each attribute value we create a child. 
            # Children are the attribute values with which we connect to other nodes.
            child = Node() 
            child.value = value 
            print('Setting child value as {}'.format(child.value))

            node.childs.append(child)  # append new child node to current node (node is the node.next of the previous call)
            child_x_ids = [x for x in x_ids if self.X[x][best_attribute_idx] == value]
            # child_x_idx = Examples_vi
            # the next of a child (attribute value) is either a class or a new attribute
            if not child_x_ids: 
                # If there are no examples, we end the branch choosing the most popular class as value
                child.next = max(set(y_in_features), key=y_in_features.count) 
                print('For child {} (in node {}) , child.next with {} is added (no more examples with that attr value) '.format(child.value, node.value, child.next.value))
                print('')
            else:
                if attribute_idxs and best_attribute_idx in attribute_idxs:
                    to_remove = attribute_idxs.index(best_attribute_idx)
                    attribute_idxs.pop(to_remove)
                # recursively call the algorithm with the reduced set of attributes
                # print('calling with child.next as {}'.format(child.next))
                child.next = self._id3_recv(child_x_ids, attribute_idxs, child.next)
                print('For child {} (in node {}) , child.next with {} is added (after recursive call)'.format(child.value, node.value, child.next.value))
        print('Children for node {} are {}'.format(node.value, [c.value for c in node.childs]))
                
        return node

    def printTree(self):
        if not self.node:
            return
        nodes = deque()
        nodes.append(self.node)
        while len(nodes) > 0:
            node = nodes.popleft()
            print('-------------')
            print('<<' + node.value + '>>')
            if node.childs:
                for child in node.childs:
                    print('{}: Child is ({})'.format(node.value, child.value))
                    nodes.append(child.next)
            elif node.next:
                print('{}: NEXT NODE {}',format(node.value, node.next))