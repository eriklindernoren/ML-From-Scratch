import sys, os
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import helper functions
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../")
from helper_functions import calculate_covariance_matrix, calculate_correlation_matrix

df = pd.read_csv(dir_path + "/../data/iris.csv")
# Change class labels from strings to numbers
df = df.replace(to_replace="setosa", value="0")
df = df.replace(to_replace="virginica", value="1")
df = df.replace(to_replace="versicolor", value="2")

# Only select data for two classes
X = df.loc[df['species'] != "2"].drop("species", axis=1).as_matrix()
y = df.loc[df['species'] != "2"]["species"].as_matrix()
X1 = df.loc[df['species'] == "0"].drop("species", axis=1).as_matrix()
X2 = df.loc[df['species'] == "1"].drop("species", axis=1).as_matrix()

# Calculate the covariances of the two class distributions
cov1 = calculate_covariance_matrix(X1, X1)
cov2 = calculate_covariance_matrix(X2, X2)
cov_tot = cov1 + cov2

# Get the means of the two class distributions
mean1 = X1.mean(0)
mean2 = X2.mean(0)
mean_diff = np.atleast_1d(mean1 - mean2)

# Calculate w as  (x1_mean - x2_mean) / (cov1 + cov2)
w = np.linalg.inv(cov_tot).dot(mean_diff)

# Project X onto w
x1 = X.dot(w)
x2 = X.dot(w)

# Plot the data
plt.scatter(x1,x2,c=y)
plt.show()