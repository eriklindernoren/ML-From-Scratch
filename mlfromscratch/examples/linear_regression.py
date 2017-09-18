import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import helper functions
from mlfromscratch.utils.data_manipulation import train_test_split, polynomial_features
from mlfromscratch.utils.data_operation import mean_squared_error
from mlfromscratch.utils import Plot
from mlfromscratch.supervised_learning import LinearRegression

def main():

    # Load temperature data
    data = pd.read_csv('mlfromscratch/data/TempLinkoping2016.txt', sep="\t")

    time = np.atleast_2d(data["time"].as_matrix()).T
    temp = np.atleast_2d(data["temp"].as_matrix()).T

    X = time        # Time. Fraction of the year [0, 1]
    y = temp[:, 0]  # Temperature. Reduce to one-dim

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    n_samples, n_features = np.shape(X)

    clf = LinearRegression()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print ("Mean squared error: %s" % (mse))

    y_pred_line = clf.predict(X)

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