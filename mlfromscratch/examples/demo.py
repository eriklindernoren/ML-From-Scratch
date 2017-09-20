from __future__ import print_function
from sklearn import datasets
import numpy as np
import math
import matplotlib.pyplot as plt

from mlfromscratch.utils import train_test_split, normalize, to_categorical, accuracy_score
from mlfromscratch.deep_learning.optimizers import Adam
from mlfromscratch.deep_learning.loss_functions import CrossEntropy
from mlfromscratch.deep_learning.activation_functions import Softmax
from mlfromscratch.utils.kernels import *
from mlfromscratch.supervised_learning import *
from mlfromscratch.deep_learning import *
from mlfromscratch.unsupervised_learning import PCA
from mlfromscratch.deep_learning.layers import Dense, Dropout, Conv2D, Flatten, Activation


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
mlp = NeuralNetwork(optimizer=Adam(), 
                    loss=CrossEntropy)
mlp.add(Dense(input_shape=(n_features,), n_units=64))
mlp.add(Activation('relu'))
mlp.add(Dense(n_units=64))
mlp.add(Activation('relu'))
mlp.add(Dense(n_units=2))   
mlp.add(Activation('softmax'))
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
mlp.fit(X_train, to_categorical(y_train), n_epochs=300, batch_size=50)
print ("- Naive Bayes")
naive_bayes.fit(X_train, y_train)
print ("- Perceptron")
perceptron.fit(X_train, to_categorical(y_train))
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
y_pred["Multilayer Perceptron"] = np.argmax(mlp.predict(X_test), axis=1)
y_pred["Perceptron"] = np.argmax(perceptron.predict(X_test), axis=1)
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


