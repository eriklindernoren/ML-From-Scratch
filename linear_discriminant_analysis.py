from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from helper_functions import calculate_covariance_matrix, calculate_correlation_matrix
import pandas as pd

df = pd.read_csv("./data/iris.csv")

# Only select data for two classes
X = df.loc[df['species'] != "versicolor"].drop("species", axis=1).as_matrix()
X_set = df.loc[df['species'] == "setosa"].drop("species", axis=1).as_matrix()
X_vir = df.loc[df['species'] == "virginica"].drop("species", axis=1).as_matrix()

# Calculate the covariances of the two class distributions
cov_set = calculate_covariance_matrix(X_set)
cov_vir = calculate_covariance_matrix(X_vir)
cov_tot = cov_set + cov_vir

# Get the means of the two class distributions
mean_set = X_set.mean(0)
mean_vir = X_vir.mean(0)
mean_diff = (mean_set - mean_vir).reshape((len(mean_set), 1))

# Calculate w as (cov1 + cov2)^(-1) * (x1_mean - x2_mean)
w = np.linalg.inv(cov_tot).dot(mean_diff)


# Change class labels from strings to numbers
df = df.replace(to_replace="setosa", value="0")
df = df.replace(to_replace="virginica", value="1")
# Only include two classes
Y = df.loc[df['species'] != "versicolor"]["species"].as_matrix()

# Project X onto w
a = X.dot(w)
b = X.dot(w)

# Plot the data
plt.scatter(a,b,c=Y)
plt.show()