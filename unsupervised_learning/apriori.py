from __future__ import division
import pandas as pd
import numpy as np
import itertools

class Apriori():
    def __init__(self, min_sup=0.3):
        self.min_sup = min_sup
        self.freq_itemsets = None

    def _get_frequent_itemsets(self, candidates, transactions):
        frequent = []
        # Find frequent items
        for itemset in candidates:
            count = 0
            for transaction in transactions:
                # If the itemset contains many items (list of ints)
                single_item = not isinstance(itemset, int)
                # If list of items and transaction contains all items
                if not single_item and self._items_in_transaction(itemset, transaction):
                    count += 1
                # If single item and transaction contains item
                if single_item and itemset in transaction:
                    count += 1
            support = count / len(transactions)
            if support >= self.min_sup:
                frequent.append(itemset)
        return frequent

    # True or false depending on the candidate has any 
    # subset with size k - 1 that is not in the frequent
    # itemset
    def _has_infrequent_itemsets(self, candidate):
        k = len(candidate)
        # Find all combinations of size k-1 in candidate
        subsets = list(itertools.combinations(candidate, k-1))
        for t in subsets:
            # t - is tuple. If size == 1 get the element
            subset = list(t) if len(t) > 1 else t[0]
            if not subset in self.freq_itemsets[-1]:
                return True
        return False


    # Joins the elements in the frequent itemset and prunes
    # resulting sets if they contain subsets that have been determined
    # to be infrequent.
    def _generate_candidates(self, freq_itemset):
        candidates = []
        for itemset1 in freq_itemset:
            for itemset2 in freq_itemset:
                # Valid if every element but the last are the same
                # and the last element in itemset1 is smaller than the last
                # in itemset2
                valid = False
                single_item = isinstance(itemset1, int)
                if single_item and itemset1 < itemset2:
                    valid = True
                elif not single_item and np.array_equal(itemset1[:-1], itemset2[:-1]) and itemset1[-1] < itemset2[-1]:
                    valid = True

                if valid:
                    # Add the last element in itemset2 to itemset1 to 
                    # create a new candidate
                    if single_item:
                        candidate = [itemset1, itemset2]
                    else:
                        candidate = itemset1 + [itemset2[-1]]
                    # Check if any subset of candidate have been determined
                    # to be infrequent
                    infrequent = self._has_infrequent_itemsets(candidate)
                    if not infrequent:
                        candidates.append(candidate)
        return candidates


    # True or false depending on each item in the itemset is also
    # in the transaction
    def _items_in_transaction(self, items, transaction):
        for item in items:
            if not item in transaction:
                return False
        return True

    # Returns the set of frequent itemsets in the list of transactions
    def find_frequent_itemsets(self, transactions):
        # Get all unique items in the transactions
        unique_items = set(item for transaction in transactions for item in transaction)
        # Get the frequent items
        self.freq_itemsets = [self._get_frequent_itemsets(unique_items, transactions)]
        while(True):
            # Generate new candidates from last added frequent itemsets
            candidates = self._generate_candidates(self.freq_itemsets[-1])
            # Get the frequent itemsets among those candidates
            freq = self._get_frequent_itemsets(candidates, transactions)

            # If we have an empty list we're done
            if not freq:
                break

            # Add them to the total list of frequent itemsets and start over
            self.freq_itemsets.append(freq)

        # Flatten the array and return every frequent itemset
        frequent_itemsets = [itemset for sublist in self.freq_itemsets for itemset in sublist]
        return frequent_itemsets




def main():
    # Demo transaction set
    # Example 2: https://en.wikipedia.org/wiki/Apriori_algorithm

    transactions = np.array([[1,2,3,4], [1,2,4], [1,2], [2,3,4], [2,3], [3,4], [2,4]])
    print "Transactions:"
    print transactions

    apriori = Apriori(min_sup=3/7)
    itemsets = apriori.find_frequent_itemsets(transactions)

    # Print the results to make sure that we have the same itemsets
    # as on wiki
    print "Frequent itemsets:"
    print itemsets

if __name__ == "__main__":
    main()
