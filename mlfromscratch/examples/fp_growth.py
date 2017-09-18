
import numpy as np
from mlfromscratch.unsupervised_learning import FPGrowth

def main():
    # Demo transaction set
    # Example:
    # https://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Frequent_Pattern_Mining/The_FP-Growth_Algorithm
    
    transactions = np.array([
        ["A", "B", "D", "E"],
        ["B", "C", "E"],
        ["A", "B", "D", "E"],
        ["A", "B", "C", "E"],
        ["A", "B", "C", "D", "E"],
        ["B", "C", "D"]
    ])

    print ("")
    print ("+---------------+")
    print ("|   FP-Growth   |")
    print ("+---------------+")
    min_sup = 3
    print ("Minimum Support: %s" % min_sup)
    print ("")
    print ("Transactions:")
    for transaction in transactions:
        print ("\t%s" % transaction)

    fp_growth = FPGrowth(min_sup=min_sup)

    print ("")
    # Get and print the frequent itemsets
    frequent_itemsets = fp_growth.find_frequent_itemsets(
        transactions, show_tree=True)

    print ("")
    print ("Frequent itemsets:")
    for itemset in frequent_itemsets:
        print ("\t%s" % itemset)
    print ("")

if __name__ == "__main__":
    main()