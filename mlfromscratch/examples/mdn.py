from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# Import helper functions
from mlfromscratch.deep_learning import NeuralNetwork
from mlfromscratch.deep_learning.layers import MDN, Dense, Activation
from mlfromscratch.deep_learning.optimizers import Adam
from mlfromscratch.deep_learning.loss_functions import MdnLoss


def main():
    # define the model
    components = 3
    optimizer = Adam()
    loss = MdnLoss(num_components=components, output_dim=1)
    clf = NeuralNetwork(optimizer=optimizer, loss=loss)
    clf.add(Dense(n_units=26, input_shape=(1, )))
    clf.add(Activation('tanh'))
    clf.add(
        MDN(input_shape=(26,), output_shape=(1,),
            num_components=components)
    )
    clf.summary(name="MDN")

    # generate 1D regression data (Bishop book, page 273).
    # Note: P(y|x) is not a nice distribution.
    # (e.g.) it has three modes for x ~= 0.5
    N = 225
    X = np.linspace(0, 1, N)
    Y = X + 0.3 * np.sin(2*3.1415926*X) + np.random.uniform(-0.1, 0.1, N)
    X, Y = Y, X
    nb = N  # full_batch
    xbatch = np.reshape(X[:nb], (nb, 1))
    ybatch = np.reshape(Y[:nb], (nb, 1))
    train_err, val_err = clf.fit(xbatch, ybatch, n_epochs=int(4e3),
                                 batch_size=N)
    plt.plot(train_err, label="Training Error")
    plt.title("Error Plot")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()

    # utility function for creating contour plot of the predictions
    n = 15
    xx = np.linspace(0, 1, n)
    yy = np.linspace(0, 1, n)
    xm, ym = np.meshgrid(xx, yy)
    loss, acc = clf.test_on_batch(
        xm.reshape(xm.size, 1),
        ym.reshape(ym.size, 1)
    )
    ypred = clf.loss_function.ypred
    plt.figure(figsize=(10, 10))
    plt.scatter(X, Y, color='g')
    plt.contour(xm, ym, np.reshape(ypred, (n, n)),
                levels=np.linspace(ypred.min(), ypred.max(), 20))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('{}-component Gaussian Mixture Model for '
              'P(y|x)'.format(components))
    plt.show()


if __name__ == "__main__":
    main()
