# Machine Learning From Scratch


###About
Python implementations of some of the foundational Machine Learning models and algorithms from scratch.

While some of the matrix operations that are implemented by hand (such as calculation of covariance matrix) are 
available in numpy I have decided to add these as well to make sure that I understand how the linear algebra is applied.
The reason the project uses scikit-learn is to evaluate the implementations on sklearn.datasets.

The purpose of this project is purely self-educational.

Feel free to [reach out](mailto:eriklindernoren@gmail.com) if you can think of ways to expand this project.


### Installation
    pip install -r requirements.txt


### Testing Implementations
    python demo.py
    python supervised_learning/multilayer_perceptron.py


##Current Implementations
####Supervised Learning:
- [Adaboost](supervised_learning/adaboost.py)
- [Decision Tree (regression and classification)](supervised_learning/decision_tree.py)
- [Gradient Boosting (regression)](supervised_learning/gradient_boosting_regressor.py)
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

####Unsupervised Learning:
- [Apriori](unsupervised_learning/apriori.py)
- [DBSCAN](unsupervised_learning/dbscan.py)
- [FP-Growth](unsupervised_learning/fp_growth.py)
- [Gaussian Mixture Model](unsupervised_learning/gaussian_mixture_model.py)
- [K-Means](unsupervised_learning/k_means.py)
- [Partitioning Around Medoids](unsupervised_learning/partitioning_around_medoids.py)
- [Principal Component Analysis](unsupervised_learning/principal_component_analysis.py)
