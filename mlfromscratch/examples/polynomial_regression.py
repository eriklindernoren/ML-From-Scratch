from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Import helper functions
from mlfromscratch.supervised_learning import PolynomialRidgeRegression
from mlfromscratch.utils import k_fold_cross_validation_sets, normalize, mean_squared_error
from mlfromscratch.utils import train_test_split, polynomial_features, Plot


def main():

    # Load temperature data
    data = pd.read_csv('mlfromscratch/data/TempLinkoping2016.txt', sep="\t")

    time = np.atleast_2d(data["time"].as_matrix()).T
    temp = data["temp"].as_matrix()

    X = time # fraction of the year [0, 1]
    y = temp

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    poly_degree = 15

    # Finding regularization constant using cross validation
    lowest_error = float("inf")
    best_reg_factor = None
    print ("Finding regularization constant using cross validation:")
    k = 10
    for reg_factor in np.arange(0, 0.1, 0.01):
        cross_validation_sets = k_fold_cross_validation_sets(
            X_train, y_train, k=k)
        mse = 0
        for _X_train, _X_test, _y_train, _y_test in cross_validation_sets:
            model = PolynomialRidgeRegression(degree=poly_degree, 
                                            reg_factor=reg_factor,
                                            learning_rate=0.001,
                                            n_iterations=10000)
            model.fit(_X_train, _y_train)
            y_pred = model.predict(_X_test)
            _mse = mean_squared_error(_y_test, y_pred)
            mse += _mse
        mse /= k

        # Print the mean squared error
        print ("\tMean Squared Error: %s (regularization: %s)" % (mse, reg_factor))

        # Save reg. constant that gave lowest error
        if mse < lowest_error:
            best_reg_factor = reg_factor
            lowest_error = mse

    # Make final prediction
    model = PolynomialRidgeRegression(degree=poly_degree, 
                                    reg_factor=best_reg_factor,
                                    learning_rate=0.001,
                                    n_iterations=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print ("Mean squared error: %s (given by reg. factor: %s)" % (lowest_error, best_reg_factor))

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')

    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.suptitle("Polynomial Ridge Regression")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()
