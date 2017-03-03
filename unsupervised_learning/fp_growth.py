from __future__ import division
import pandas as pd
import numpy as np
import itertools
import sys

class FPTreeNode():
    def __init__(self, item=None, support=1, parent=None):
        self.item = item
        self.support = support
        self.parent = None
        self.children = {}
        self.frequent_itemsets = []

class FPGrowth():
    def __init__(self, min_sup=0.3):
        self.min_sup = min_sup
        self.tree_root = None
        self.prefixes = {}
        self.count = 0
        self.frequent_itemsets = []

    # Count the number of transactions that contains item.
    def _calculate_support(self, item, transactions):
        count = 0
        for transaction in transactions:
            if item in transaction:
                count += 1
        support = count
        return support

    # Returns a set of frequent items. An item is determined to 
    # be frequent if there are atleast min_sup transactions that contains
    # it.
    def _get_frequent_items(self, transactions):
        # Get all unique items in the transactions
        unique_items = set(item for transaction in transactions for item in transaction)
        items = []
        for item in unique_items:
            sup = self._calculate_support(item, transactions)
            if sup >= self.min_sup:
                items.append([item, sup])
        # Sort by support - Highest to lowest
        items.sort(key=lambda item: item[1], reverse=True)
        frequent_items = [[el[0]] for el in items]
        # Only return the items
        return frequent_items


    # Recursive method which adds nodes to the tree.
    def _insert_tree(self, node, children):
        if not children:
            return

        # Create new node as the first item in children list
        child_item = children[0]
        child = FPTreeNode(item=child_item, parent=node)
        # If parent already contains item => increase the support
        if child_item in node.children:
            node.children[child.item].support += 1
        else:
            node.children[child.item] = child

        # Execute _insert_tree on the rest of the children list
        # from the new node
        self._insert_tree(node.children[child.item], children[1:])


    def _construct_tree(self, transactions, frequent_items=None):
        if not frequent_items:
            # Get a list of frequent 1D items and support ([item, support]) sorted by support
            frequent_items = self._get_frequent_items(transactions)
        unique_frequent_items = list(set(item for itemset in frequent_items for item in itemset))
        # Construct the root of the FP Growth tree
        root = FPTreeNode()
        for transaction in transactions:
            # Remove items that are not frequent according to unique_frequent_items
            transaction = [item for item in transaction if item in unique_frequent_items]
            transaction.sort(key=lambda item: frequent_items.index([item]))
            self._insert_tree(root, transaction)

        return root

    # Recursive method which prints the FP Growth Tree
    def print_tree(self, node=None, indent_times=0):
        indent = "    " * indent_times
        print "%s%s:%s" % (indent, node.item, node.support)
        for child_key in node.children:
            child = node.children[child_key]
            self.print_tree(child, indent_times+1)

    # Makes sure that the first item in itemset
    # is a child of node and that every following item
    # in itemset is reachable via that path
    def _is_prefix(self, itemset, node):
        for item in itemset:
            if not item in node.children:
                return False
            node = node.children[item]
        return True

    # Recursive method that adds prefixes to the itemset by
    # traversing the FP Growth Tree
    def _determine_prefixes(self, itemset, node, prefixes=None):
        if not prefixes:
            prefixes = []

        for child_key in node.children:
            child = node.children[child_key]
            # If the first item in the itemset is the same as the
            # child of node => check if node is a prefix to itemset
            if child.item == itemset[0] and node.item:
                if self._is_prefix(itemset, node):
                    itemset_key = self._get_itemset_key(itemset)
                    if not itemset_key in self.prefixes:
                        self.prefixes[itemset_key] = []
                    self.prefixes[itemset_key] += [{"prefix": prefixes, "support": child.support}]

            # Recursive chall with child as new node. Add the child item as potential
            # prefix.
            self._determine_prefixes(itemset, child, prefixes + [child.item])

    # Determines the look of the hashmap key for self.prefixes
    # List of more strings than one gets joined by '-'
    def _get_itemset_key(self, itemset):
        if len(itemset) > 1:
            itemset_key = "-".join(itemset)
        else:
            itemset_key = str(itemset[0])
        return itemset_key

    def _determine_frequent_itemsets(self, conditional_database, suffix):
        # Calculate new frequent items from the conditional database
        # of suffix
        frequent_items = self._get_frequent_items(conditional_database)

        cond_tree = None

        if suffix:
            cond_tree = self._construct_tree(conditional_database)
            # Output new frequent itemset as the suffix added to the frequent items
            self.frequent_itemsets += [el + suffix for el in frequent_items]

        # Determine larger frequent itemset by checking prefixes
        # of the frequent itemsets
        self.prefixes = {}
        for itemset in frequent_items:
            # If no suffix (first run)
            if not cond_tree:
                cond_tree = self.tree_root
            # Determine prefixes to itemset
            self._determine_prefixes(itemset, cond_tree)
            conditional_database = []
            itemset_key = self._get_itemset_key(itemset)
            # Build conditional database
            if itemset_key in self.prefixes:
                for el in self.prefixes[itemset_key]:
                    # If support = 4 => add 4 of the corresponding prefix set
                    for _ in range(el["support"]):
                        conditional_database.append(el["prefix"])
                # Create new suffix
                new_suffix = itemset + suffix if suffix else itemset
                self._determine_frequent_itemsets(conditional_database, suffix=new_suffix)
    

    def find_frequent_itemsets(self, transactions, suffix=None, show_tree=False):
        self.transactions = transactions

        # Build the FP Growth Tree
        self.tree_root = self._construct_tree(transactions)
        if show_tree:
            print "FP-Growth Tree:"
            self.print_tree(self.tree_root)

        self._determine_frequent_itemsets(transactions, suffix=None)

        return self.frequent_itemsets
            


def main():
    # Demo transaction set
    # Example: https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Frequent_Pattern_Mining/The_FP-Growth_Algorithm
    transactions = np.array([
        ["A","B","D","E"],
        ["B","C","E"],
        ["A","B","D","E"],
        ["A","B","C","E"],
        ["A","B","C","D","E"],
        ["B","C","D"]
        ])

    print "- FP-Growth -"
    min_sup = 3
    print "Minimum - support: %s" % min_sup
    print "Transactions:"
    for transaction in transactions:
        print "\t%s" % transaction
    print
    fp_growth = FPGrowth(min_sup=min_sup)

    # Get and print the frequent itemsets
    frequent_itemsets = fp_growth.find_frequent_itemsets(transactions, show_tree=True)
    print
    print "Frequent itemsets:"
    for itemset in frequent_itemsets:
        print "\t%s" % itemset


if __name__ == "__main__":
    main()
