from __future__ import division
import numpy as np
import pandas as pd
import math, sys

df = pd.read_csv("./data/iris.csv")

# Shuffle the samples
df = df.sample(frac=1).reset_index(drop=True)

# Use the last 100 samples as test data
df_train = df.iloc[:-100]
df_test = df.iloc[-100:]

classes = df["species"].unique()

class_map = {
	0: "setosa",
	1: "versicolor",
	2: "virginica"
}

# Gaussian probability distribution
def calculate_probability(mean, var, x):
	coeff = (1.0 / (math.sqrt((2.0*math.pi) * var)))
	exponent = math.exp(-(math.pow(x-mean,2)/(2*var)))
	return coeff * exponent

mean_var = []
def determine_mean_and_variance():
	# Calculate the mean and variance of each feature for each class
	for i in range(len(classes)):
		c = class_map[i]
		# Only select the rows where the species equals the given class
		class_df = df_train.loc[df_train['species'] == c].drop("species", axis=1)
		# Add the mean and variance for each feature
		mean_var.append([])
		for col in class_df:
			mean = class_df[col].mean()
			var = class_df[col].var()
			mean_var[i].append([mean, var])
	# Return the list
	return mean_var

# Calculate the prior of class c (samples where class == c / total number of samples)
def calculate_prior(c):
	# Only select the rows where the species equals the given class
	class_df = df_train.loc[df_train['species'] == c].drop("species", axis=1)
	n_class_instances = class_df.shape[0]
	n_total_instances = df_train.shape[0]
	return n_class_instances / n_total_instances

# Compare the sample to the gaussian distribution of each class
# Choose the class that has the highest probability of the sample 
# being a member of that distribution
def classify(sample):
	results = []
	# Go through list of classes
	for i in range(len(classes)):
		c = class_map[i]
		results.append([])
		# Add the prior as the first probability
		prior = calculate_prior(c)
		probs = np.array([prior])
		# Add the additional probabilities
		for j in range(len(mean_var[i])):
			sample_feature = sample[j]
			mean = mean_var[i][j][0]
			var = mean_var[i][j][1]
			# Determine probability of sample belonging the class 'c' by the 
			# sample's distance to the class distribution 
			prob = calculate_probability(mean, var, sample_feature)
			probs = np.append(probs, prob)
		probability = np.prod(probs)
		results[i] = probability
	# Get the largest probability and return the class corresponding
	# to that probability
	index_of_max = np.argmax(results)
	max_value = results[index_of_max]

	return class_map[index_of_max]

# Determine the mean and variance of the features in the 
# training set
determine_mean_and_variance()

# Test the model
x_test = df_test.drop("species", axis=1).as_matrix()
y_test = df_test["species"].as_matrix()

n_test_samples = len(y_test)

correct = 0
for i in range(n_test_samples):
	sample = x_test[i, :]
	y_pred = classify(sample)
	y_true = y_test[i]
	# print
	# print "Prediction:", y_pred
	# print "True:", y_true
	if y_pred == y_true:
		correct += 1

accuracy = correct / n_test_samples
print "Accuracy:", accuracy
