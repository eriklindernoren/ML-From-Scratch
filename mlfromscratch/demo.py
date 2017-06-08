from __future__ import print_function
import sys, os
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import helper functions
from mlfromscratch.utils.data_manipulation import train_test_split, normalize
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.utils.kernels import *
# Import ML models
from mlfromscratch.supervised_learning import *
# Import PCA
from mlfromscratch.unsupervised_learning import PCA


print ("+-------------------------------------------+")
print ("|                                           |")
print ("|       Machine Learning From Scratch       |")
print ("|                                           |")
print ("+-------------------------------------------+")


# ...........
#  LOAD DATA
# ...........
data = datasets.load_digits()
digit1 = 1
digit2 = 8
idx = np.append(np.where(data.target == digit1)[0], np.where(data.target == digit2)[0])
y = data.target[idx]
# Change labels to {0, 1}
y[y == digit1] = 0
y[y == digit2] = 1
X = data.data[idx]
X = normalize(X)

print ("Dataset: The Digit Dataset (digits %s and %s)" % (digit1, digit2))

# ..........................
#  DIMENSIONALITY REDUCTION
# ..........................
pca = PCA()
X = pca.transform(X, n_components=5) # Reduce to 5 dimensions


# ..........................
#  TRAIN / TEST SPLIT
# ..........................
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# Rescaled labels {-1, 1}
rescaled_y_train = 2*y_train - np.ones(np.shape(y_train))
rescaled_y_test = 2*y_test - np.ones(np.shape(y_test))

# .......
#  SETUP
# .......
adaboost = Adaboost(n_clf = 8)
naive_bayes = NaiveBayes()
knn = KNN(k=4)
logistic_regression = LogisticRegression()
mlp = MultilayerPerceptron(n_hidden=20, n_iterations=20000, learning_rate=0.1)
perceptron = Perceptron()
decision_tree = ClassificationTree()
random_forest = RandomForest(n_estimators=50)
support_vector_machine = SupportVectorMachine()
lda = LDA()
gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=.9, max_depth=2)
xgboost = XGBoost(n_estimators=50, learning_rate=0.5)

# ........
#  TRAIN
# ........
print ("Training:")
print ("\tAdaboost")
adaboost.fit(X_train, rescaled_y_train)
print ("\tDecision Tree")
decision_tree.fit(X_train, y_train)
print ("\tGradient Boosting")
gbc.fit(X_train, y_train)
print ("\tLDA")
lda.fit(X_train, y_train)
print ("\tLogistic Regression")
logistic_regression.fit(X_train, y_train)
print ("\tMultilayer Perceptron")
mlp.fit(X_train, y_train)
print ("\tNaive Bayes")
naive_bayes.fit(X_train, y_train)
print ("\tPerceptron")
perceptron.fit(X_train, y_train)
print ("\tRandom Forest")
random_forest.fit(X_train, y_train)
print ("\tSupport Vector Machine")
support_vector_machine.fit(X_train, rescaled_y_train)
print ("\tXGBoost")
xgboost.fit(X_train, y_train)



# .........
#  PREDICT
# .........
y_pred = {}
y_pred["Adaboost"] = adaboost.predict(X_test)
y_pred["Gradient Boosting"] = gbc.predict(X_test)
y_pred["Naive Bayes"] = naive_bayes.predict(X_test)
y_pred["K Nearest Neighbors"] = knn.predict(X_test, X_train, y_train)
y_pred["Logistic Regression"] = logistic_regression.predict(X_test)
y_pred["LDA"] = lda.predict(X_test)
y_pred["Multilayer Perceptron"] = mlp.predict(X_test)
y_pred["Perceptron"] = perceptron.predict(X_test)
y_pred["Decision Tree"] = decision_tree.predict(X_test)
y_pred["Random Forest"] = random_forest.predict(X_test)
y_pred["Support Vector Machine"] = support_vector_machine.predict(X_test)
y_pred["XGBoost"] = xgboost.predict(X_test)

# ..........
#  ACCURACY
# ..........

print ("Accuracy:")
for clf in y_pred:
	if clf == "Adaboost" or clf == "Support Vector Machine":
		print ("\t%-23s: %.5f" %(clf, accuracy_score(rescaled_y_test, y_pred[clf])))
	else:
		print ("\t%-23s: %.5f" %(clf, accuracy_score(y_test, y_pred[clf])))

# .......
#  PLOT
# .......
plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
plt.ylabel("Principal Component 2")
plt.xlabel("Principal Component 1")
plt.title("The Digit Dataset (digits %s and %s)" % (digit1, digit2))
plt.show()


