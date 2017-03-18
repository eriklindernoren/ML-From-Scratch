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
    + [Classification](#classification)
    + [Clustering](#clustering)
    + [Association Analysis](#association-analysis)
  * [Current Implementations](#current-implementations)
    + [Supervised Learning](#supervised-learning)
    + [Unsupervised Learning](#unsupervised-learning)
    
## Usage
### Installation
    $ pip install -r requirements.txt

### Classification
    $ python supervised_learning/multilayer_perceptron.py

<p align="center">
    <img src="http://eriklindernoren.se/images/mlp.png">
</p>
<p align="center">
    Figure: Classification of the digit dataset using MLP.
</p>

### Clustering
    $ python unsupervised_learning/dbscan.py
   
<p align="center">
    <img src="http://eriklindernoren.se/images/dbs.png">
</p>
<p align="center">
    Figure: Clustering of the moons dataset using DBSCAN.
</p>

### Association Analysis
    $ python unsupervised_learning/apriori.py 
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
- [Adaboost](supervised_learning/adaboost.py)
- [Decision Tree](supervised_learning/decision_tree.py)
- [Gradient Boosting](supervised_learning/gradient_boosting.py)
- [K Nearest Neighbors](supervised_learning/k_nearest_neighbors.py)
- [Linear Discriminant Analysis](supervised_learning/linear_discriminant_analysis.py)
- [Linear Regression](supervised_learning/linear_regression.py)
- [Logistic Regression](supervised_learning/logistic_regression.py)
- [Multi-class Linear Discriminant Analysis](supervised_learning/multi_class_lda.py)
- [Multilayer Perceptron](supervised_learning/multilayer_perceptron.py)
- [Naive Bayes](supervised_learning/naive_bayes.py)
- [Perceptron](supervised_learning/perceptron.py)
- [Random Forest](supervised_learning/random_forest.py)
- [Ridge Regression](supervised_learning/ridge_regression.py)
- [Support Vector Machine](supervised_learning/support_vector_machine.py)
- [XGBoost](supervised_learning/xgboost.py)

### Unsupervised Learning
- [Apriori](unsupervised_learning/apriori.py)
- [DBSCAN](unsupervised_learning/dbscan.py)
- [FP-Growth](unsupervised_learning/fp_growth.py)
- [Gaussian Mixture Model](unsupervised_learning/gaussian_mixture_model.py)
- [K-Means](unsupervised_learning/k_means.py)
- [Partitioning Around Medoids](unsupervised_learning/partitioning_around_medoids.py)
- [Principal Component Analysis](unsupervised_learning/principal_component_analysis.py)
