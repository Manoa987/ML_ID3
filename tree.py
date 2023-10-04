from random import random
import pandas as pd
from typing import List

from node import Node
from attribute import Attribute


class DecisionTree:

    def __init__(self, max_depth, min_samples):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def __build_tree(self, df: pd.DataFrame, attributes: List[Attribute], target_attr: Attribute, d: int) -> Node:
        if len(df) == 0:
            return Node(value='?',attribute=None)

        if len(df[target_attr.label].unique()) == 1 or d > self.max_depth or len(df) <= self.min_samples:
            return Node(value=df[target_attr.label].mode()[0],attribute=None)

        else:
            gains = {}
            for attribute in attributes:
                gains[attribute] = attribute.get_gain(df, target_attr)

            best_attr = max(gains, key=gains.get)

            newNode = Node(value='?',attribute=best_attr)
            remaining_attributes = list(attributes)
            remaining_attributes.remove(best_attr)

            for value in best_attr.values:
                subtree = self.__build_tree(df[df[newNode.attribute.label] == value], remaining_attributes, target_attr, d + 1)


                if (not subtree.children) and subtree.value == '?':
                # if type(subtree) is TerminalNode and subtree.value == '?':
                    subtree = Node(value=df[target_attr.label].mode()[0],attribute=None)
                newNode.children[value] = subtree

            return newNode

    def train(self, samples: pd.DataFrame, attributes: List[Attribute], target_attr: Attribute):
        self.root: Node = self.__build_tree(samples, attributes, target_attr, 0)


    def __evaluate(self, sample: dict, node: Node) -> str:
        if not node.children:
            return node.value
        else:
            value = sample[node.attribute.label]
            children = node.children[value]
            return self.__evaluate(sample, children)

    def evaluate(self, sample: dict) -> str:
        return self.__evaluate(sample, self.root)

    
    def __get_node_amount(self, node: Node) -> int:
        if not node.children:
            return 1

        amount = 1
        for child in node.children.values():
            amount += self.__get_node_amount(child)

        return amount

    def get_node_amount(self) -> int:
        return self.__get_node_amount(self.root)
    
    