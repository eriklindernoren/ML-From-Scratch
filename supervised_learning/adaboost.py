from __future__ import division
import numpy as np
from sklearn.datasets import make_gaussian_quantiles
import matplotlib.pyplot as plt
import math, sys

# Construct dataset
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=1)
x_train = np.concatenate((X1, X2))
y_train = np.concatenate((y1, - y2 + 1))
y_train = 2*y_train - np.ones(np.shape(y_train))

n_samples = len(y_train)
n_features = len(x_train[0])

# Initialize weights to 1/N
w = np.ones(n_samples)*(1/n_features)

# Set number of weak classifiers
n_clf = 6

# clf = [threshold, polarity, feature_index, alpha]
clfs = np.ones((4,n_clf))

# Iterate through classifiers
for c in range(n_clf):
    # Initial values
    err_min = 1
    polarity = 0
    threshold = 0
    feature_index = 0
    # Iterate throught every sample for each feature
    for n in range(n_features):
        for t in range(n_samples):
            err = 0
            # Select feature as threshold
            tao = x_train[t, n]
            # Iterate through all samples and measure the corresponding features
            # against the selected threshold and see if the threshold can help predict
            # y
            for m in range(n_samples):
                x = x_train[m, n]
                y = y_train[m]
                h = 1.0
                p = 1
                if p*x < p*tao:
                    h = -1.0
                err = err + w[m]*(y != h)
            # E.g err = 0.8 => (1 - err) = 0.2
            # We flip the error and polarity
            if err > 0.5 and err < 1:
                err = 1 - err
                p = -1
            # If this threshold resulted in the smallest error we save the
            # the configuration
            if err < err_min:
                polarity = p
                threshold = tao
                feature_index = n
                err_min = err
    # Calculate the alpha which is used to update the sample weights
    alpha = 0.5*math.log((1.0001-err_min)/(err_min + 0.0001))
    # Save the classifier configuration
    clfs[:4,c] = [threshold, polarity, feature_index, alpha]
    # Iterate through samples and update weights 
    # Large weight => hard sample to classify
    for m in range(n_samples):
        h = 1.0
        x = x_train[m, feature_index]
        y = y_train[m]
        if polarity*x < polarity*threshold:
            h = -1.0
        w[m] = w[m]*math.exp(-alpha*y*h)
    # Renormalize the weight vector
    w = w * (1/np.sum(w))

print clfs

# Test data
x_test = np.array([[0.3, 0.98],[3.4, 1.74]])
y_test = np.array([-1, 1])

correct = 0
# Iterate through each test sample and classify by commitee
for i in range(len(y_test)):
    s = 0
    for c in range(n_clf):
        # Get classifier parameters
        clf = clfs[:, c]
        threshold = clf[0]
        polarity = clf[1]
        feature_index = int(clf[2])
        alpha = clf[3]
        x = x_test[i, feature_index]
        h = 1
        if polarity*x < polarity*threshold:
            h = -1
        # Weight prediction by classifiers proficiency
        s += alpha*h
    y = y_test[i]
    # if s < 0 => pred = -1, else pred = 1
    correct += (np.sign(s) == y)

accuracy = correct / len(y_test)
print "Accuracy:", accuracy

plt.scatter(x_train[:,0], x_train[:,1])
plt.show()
