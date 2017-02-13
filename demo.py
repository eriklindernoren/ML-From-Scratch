import sys, os
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
from helper_functions import train_test_split, accuracy_score
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

# ......
#  DATA
# ......
df = pd.read_csv(dir_path + "/data/iris.csv")
df = df.replace(to_replace="virginica", value="0")
df = df.replace(to_replace="versicolor", value="1")
# Only select data for two classes
X = df.loc[df['species'] != "setosa"].drop("species", axis=1).as_matrix()
y = df.loc[df['species'] != "setosa"]["species"].as_matrix()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# Rescale class labels for Adaboost to {-1, 1}
ada_y_train = 2*y_train - np.ones(np.shape(y_train))
ada_y_test = 2*y_test - np.ones(np.shape(y_test))

# .......
#  SETUP
# .......
adaboost = Adaboost(n_clf = 8)
naive_bayes = NaiveBayes()
knn = KNN(k=4)
logistic_regression = LogisticRegression()
mlp = MultilayerPerceptron(n_hidden=5)
perceptron = Perceptron()

# .......
#  TRAIN
# .......
adaboost.fit(x_train, ada_y_train)
naive_bayes.fit(x_train, y_train)
logistic_regression.fit(x_train, y_train)
mlp.fit(x_train, y_train, n_iterations=4000, learning_rate=0.01)
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
		print "\t%s: %s" %(clf, accuracy_score(ada_y_test, y_pred[clf]))
	else:
		print "\t%s: %s" %(clf, accuracy_score(y_test, y_pred[clf]))

# .......
#  PLOT
# .......
pca = PCA(n_components=2)
X_transformed = pca.transform(x_test)
x1 = X_transformed[:,0]
x2 = X_transformed[:,1]
plt.scatter(x1,x2,c=y_test)
plt.show()


