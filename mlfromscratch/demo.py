from __future__ import print_function
import sys, os
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlfromscratch.utils.data_manipulation import train_test_split, normalize, categorical_to_binary
from mlfromscratch.utils.data_operation import accuracy_score
from mlfromscratch.utils.optimizers import GradientDescent_
from mlfromscratch.utils.activation_functions import Softmax
from mlfromscratch.utils.kernels import *
from mlfromscratch.supervised_learning import *
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

n_samples, n_features = np.shape(X)

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
mlp = MultilayerPerceptron(n_iterations=2000, optimizer=GradientDescent_(0.001, 0.4), batch_size=50)
mlp.add(DenseLayer(n_inputs=n_features, n_units=64))
mlp.add(DenseLayer(n_inputs=64, n_units=64))
mlp.add(DenseLayer(n_inputs=64, n_units=2, activation_function=Softmax))   
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
print ("- Adaboost")
adaboost.fit(X_train, rescaled_y_train)
print ("- Decision Tree")
decision_tree.fit(X_train, y_train)
print ("- Gradient Boosting")
gbc.fit(X_train, y_train)
print ("- LDA")
lda.fit(X_train, y_train)
print ("- Logistic Regression")
logistic_regression.fit(X_train, y_train)
print ("- Multilayer Perceptron")
mlp.fit(X_train, y_train)
print ("- Naive Bayes")
naive_bayes.fit(X_train, y_train)
print ("- Perceptron")
perceptron.fit(X_train, y_train)
print ("- Random Forest")
random_forest.fit(X_train, y_train)
print ("- Support Vector Machine")
support_vector_machine.fit(X_train, rescaled_y_train)
print ("- XGBoost")
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
    # Rescaled {-1 1}
    if clf == "Adaboost" or clf == "Support Vector Machine":
        print ("\t%-23s: %.5f" %(clf, accuracy_score(rescaled_y_test, y_pred[clf])))
    # Categorical
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


