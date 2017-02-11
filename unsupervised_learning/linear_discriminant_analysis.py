from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../")
from helper_functions import calculate_covariance_matrix, calculate_correlation_matrix
import pandas as pd

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
cov1 = calculate_covariance_matrix(X1)
cov2 = calculate_covariance_matrix(X2)
cov_tot = cov1 + cov2

# Get the means of the two class distributions
mean1 = X1.mean(0)
mean2 = X2.mean(0)
mean_diff = (mean1 - mean2).reshape((len(mean1), 1))

# Calculate w as (cov1 + cov2)^(-1) * (x1_mean - x2_mean)
w = np.linalg.inv(cov_tot).dot(mean_diff)

# Project X onto w
a = X.dot(w)
b = X.dot(w)

# Plot the data
plt.scatter(a,b,c=y)
plt.show()