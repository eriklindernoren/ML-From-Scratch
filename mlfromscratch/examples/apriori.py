from __future__ import division, print_function
import numpy as np

from mlfromscratch.unsupervised_learning import Apriori

def main():
    # Demo transaction set
    # Example 2: https://en.wikipedia.org/wiki/Apriori_algorithm
    transactions = np.array([[1, 2, 3, 4], [1, 2, 4], [1, 2], [2, 3, 4], [2, 3], [3, 4], [2, 4]])
    print ("+-------------+")
    print ("|   Apriori   |")
    print ("+-------------+")
    min_sup = 0.25
    min_conf = 0.8
    print ("Minimum Support: %.2f" % (min_sup))
    print ("Minimum Confidence: %s" % (min_conf))
    print ("Transactions:")
    for transaction in transactions:
        print ("\t%s" % transaction)

    apriori = Apriori(min_sup=min_sup, min_conf=min_conf)

    # Get and print the frequent itemsets
    frequent_itemsets = apriori.find_frequent_itemsets(transactions)
    print ("Frequent Itemsets:\n\t%s" % frequent_itemsets)

    # Get and print the rules
    rules = apriori.generate_rules(transactions)
    print ("Rules:")
    for rule in rules:
        print ("\t%s -> %s (support: %.2f, confidence: %s)" % (rule.antecedent, rule.concequent, rule.support, rule.confidence,))


if __name__ == "__main__":
    main()