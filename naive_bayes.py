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

# Unique values in last column
classes = df.ix[:,-1].unique()

mean_var = []
def determine_class_distributions():
	# Calculate the mean and variance of each feature for each class
	for i in range(len(classes)):
		c = classes[i]
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

# Gaussian probability distribution
def calculate_probability(mean, var, x):
	coeff = (1.0 / (math.sqrt((2.0*math.pi) * var)))
	exponent = math.exp(-(math.pow(x-mean,2)/(2*var)))
	return coeff * exponent

# Calculate the prior of class c (samples where class == c / total number of samples)
def calculate_prior(c):
	# Only select the rows where the species equals the given class
	class_df = df_train.loc[df_train['species'] == c].drop("species", axis=1)
	n_class_instances = class_df.shape[0]
	n_total_instances = df_train.shape[0]
	return n_class_instances / n_total_instances

# Classify using Bayes Rule, P(Y|X) = P(X|Y)*P(Y)/P(X)
# P(X|Y) - Probability. Gaussian distribution (given by calculate_probability)
# P(Y) - Prior (given by calculate_prior)
# P(X) - Scales the posterior to the range 0 - 1 (ignored)
# Classify the sample as the class that results in the largest P(Y|X) (posterior)
def classify(sample):
	posteriors = []
	# Go through list of classes
	for i in range(len(classes)):
		c = classes[i]
		prior = calculate_prior(c)
		posterior = prior
		# multiply with the additional probabilties
		# Naive assumption (independence):
		# P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
		for j in range(len(mean_var[i])):
			sample_feature = sample[j]
			mean = mean_var[i][j][0]
			var = mean_var[i][j][1]
			# Determine P(x|Y)
			prob = calculate_probability(mean, var, sample_feature)
			# Multiply with the rest
			posterior *= prob
		# Total probability = P(Y)*P(x1|Y)*P(x2|Y)*...*P(xN|Y)
		posteriors.append(posterior)
	# Get the largest probability and return the class corresponding
	# to that probability
	index_of_max = np.argmax(posteriors)
	max_value = posteriors[index_of_max]

	return classes[index_of_max]

# Determine the mean and the variance of the different
# class distributions
determine_class_distributions()

# Test the model
x_test = df_test.ix[:,:-1].as_matrix()
y_test = df_test.ix[:,-1].as_matrix()

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
