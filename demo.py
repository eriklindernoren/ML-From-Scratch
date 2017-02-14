import sys, os
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
from helper_functions import train_test_split, accuracy_score, normalize
# Import ML models
sys.path.insert(0, dir_path + "/supervised_learning")
from adaboost import Adaboost
from naive_bayes import NaiveBayes
from k_nearest_neighbors import KNN
from multilayer_perceptron import MultilayerPerceptron
from logistic_regression import LogisticRegression
from perceptron import Perceptron
# Import PCA
sys.path.insert(0, dir_path + "/unsupervised_learning")
from principal_component_analysis import PCA

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

# ..........................
#  DIMENSIONALITY REDUCTION
# ..........................
pca = PCA(n_components=5) # Reduce to 5 dimensions
X = pca.transform(X)
X = normalize(X)

# ..........................
#  TRAIN / TEST SPLIT
# ..........................
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Rescale label for Adaboost to {-1, 1}
ada_y_train = 2*y_train - np.ones(np.shape(y_train))
ada_y_test = 2*y_test - np.ones(np.shape(y_test))

# .......
#  SETUP
# .......
adaboost = Adaboost(n_clf = 8)
naive_bayes = NaiveBayes()
knn = KNN(k=4)
logistic_regression = LogisticRegression()
mlp = MultilayerPerceptron(n_hidden=20)
perceptron = Perceptron()

# ........
#  TRAIN
# ........
print "Training:"
print "\tAdaboost"
adaboost.fit(x_train, ada_y_train)
print "\tNaive Bayes"
naive_bayes.fit(x_train, y_train)
print "\tLogistic Regression"
logistic_regression.fit(x_train, y_train)
print "\tMultilayer Perceptron"
mlp.fit(x_train, y_train, n_iterations=20000, learning_rate=0.1)
print "\tPerceptron"
perceptron.fit(x_train, y_train)

# .........
#  PREDICT
# .........
y_pred = {}
y_pred["Adaboost"] = adaboost.predict(x_test)
y_pred["Naive Bayes"] = naive_bayes.predict(x_test)
y_pred["K Nearest Neighbors"] = knn.predict(x_test, x_train, y_train)
y_pred["Logistic Regression"] = logistic_regression.predict(x_test)
y_pred["Multilayer Perceptron"] = mlp.predict(x_test)
y_pred["Perceptron"] = perceptron.predict(x_test)

# ..........
#  ACCURACY
# ..........
print "Accuracy:"
for clf in y_pred:
	if clf == "Adaboost":
		print "\t%-23s: %.5f" %(clf, accuracy_score(ada_y_test, y_pred[clf]))
	else:
		print "\t%-23s: %.5f" %(clf, accuracy_score(y_test, y_pred[clf]))

# .......
#  PLOT
# .......
pca = PCA(n_components=3)
X_3d = pca.transform(x_test)
x1 = X_3d[:,0]
x2 = X_3d[:,1]
x3 = X_3d[:,2]
plt.scatter(x1,x2,c=y_test)
plt.show()


