# Machine Learning From Scratch

## About
Python implementations of some of the fundamental Machine Learning models and algorithms from scratch.

The purpose of this project is not to produce as optimized and computationally efficient algorithms as possible 
but rather to present the inner workings of them in a transparent way.
The reason the project uses scikit-learn is to evaluate the implementations on sklearn.datasets.

Feel free to [reach out](mailto:eriklindernoren@gmail.com) if you can think of ways to expand this project.

## Table of Contents
- [Machine Learning From Scratch](#machine-learning-from-scratch)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Usage](#usage)
    + [Installation](#installation)
    + [Regression](#regression)
    + [Classification](#classification)
    + [Clustering](#clustering)
    + [Association Analysis](#association-analysis)
  * [Current Implementations](#current-implementations)
    + [Supervised Learning](#supervised-learning)
    + [Unsupervised Learning](#unsupervised-learning)
    
## Usage
### Installation
    $ pip install mlfs
or
    ```$ python setup.py install```

### Regression
    $ python mlfs/supervised_learning/linear_regression.py

<p align="center">
    <img src="http://eriklindernoren.se/images/pr4.png" width="640"\>
</p>
<p align="center">
    Figure: Polynomial regression of temperature data in Linköping, Sweden 2016.
</p>

### Classification
    $ python mlfs/supervised_learning/multilayer_perceptron.py

<p align="center">
    <img src="http://eriklindernoren.se/images/mlp3.png" width="640">
</p>
<p align="center">
    Figure: Classification of the digit dataset using MLP.
</p>

### Clustering
    $ python mlfs/unsupervised_learning/dbscan.py
   
<p align="center">
    <img src="http://eriklindernoren.se/images/dbscan3.png" width="640">
</p>
<p align="center">
    Figure: Clustering of the moons dataset using DBSCAN.
</p>

### Association Analysis
    $ python mlfs/unsupervised_learning/apriori.py 
    - Apriori -
    Minimum - support: 0.25, confidence: 0.8
    Transactions:
        [1, 2, 3, 4]
        [1, 2, 4]
        [1, 2]
        [2, 3, 4]
        [2, 3]
        [3, 4]
        [2, 4]
    Frequent Itemsets:
        [1, 2, 3, 4, [1, 2], [1, 4], [2, 3], [2, 4], [3, 4], [1, 2, 4], [2, 3, 4]]
    Rules:
        1 -> 2 (support: 0.43, confidence: 1.0)
        4 -> 2 (support: 0.57, confidence: 0.8)
        [1, 4] -> 2 (support: 0.29, confidence: 1.0)


## Current Implementations
### Supervised Learning
- [Adaboost](mlfs/supervised_learning/adaboost.py)
- [Bayesian Regression](mlfs/supervised_learning/bayesian_regression.py)
- [Decision Tree](mlfs/supervised_learning/decision_tree.py)
- [Gradient Boosting](mlfs/supervised_learning/gradient_boosting.py)
- [K Nearest Neighbors](mlfs/supervised_learning/k_nearest_neighbors.py)
- [Linear Discriminant Analysis](mlfs/supervised_learning/linear_discriminant_analysis.py)
- [Linear Regression](mlfs/supervised_learning/linear_regression.py)
- [Logistic Regression](mlfs/supervised_learning/logistic_regression.py)
- [Multi-class Linear Discriminant Analysis](mlfs/supervised_learning/multi_class_lda.py)
- [Multilayer Perceptron](mlfs/supervised_learning/multilayer_perceptron.py)
- [Naive Bayes](mlfs/supervised_learning/naive_bayes.py)
- [Perceptron](mlfs/supervised_learning/perceptron.py)
- [Random Forest](mlfs/supervised_learning/random_forest.py)
- [Ridge Regression](mlfs/supervised_learning/ridge_regression.py)
- [Support Vector Machine](mlfs/supervised_learning/support_vector_machine.py)
- [XGBoost](mlfs/supervised_learning/xgboost.py)

### Unsupervised Learning
- [Apriori](mlfs/unsupervised_learning/apriori.py)
- [DBSCAN](mlfs/unsupervised_learning/dbscan.py)
- [FP-Growth](mlfs/unsupervised_learning/fp_growth.py)
- [Gaussian Mixture Model](mlfs/unsupervised_learning/gaussian_mixture_model.py)
- [K-Means](mlfs/unsupervised_learning/k_means.py)
- [Partitioning Around Medoids](mlfs/unsupervised_learning/partitioning_around_medoids.py)
- [Principal Component Analysis](mlfs/unsupervised_learning/principal_component_analysis.py)
