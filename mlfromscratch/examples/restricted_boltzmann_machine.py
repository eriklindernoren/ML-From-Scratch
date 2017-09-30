import logging

import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

from mlfromscratch.unsupervised_learning import RBM

logging.basicConfig(level=logging.DEBUG)

def main():

    mnist = fetch_mldata('MNIST original')

    X = mnist.data / 255.0
    y = mnist.target

    # Select the samples of the digit 2
    X = X[y == 2]

    # Limit dataset to 500 samples
    idx = np.random.choice(range(X.shape[0]), size=500, replace=False)
    X = X[idx]

    rbm = RBM(n_hidden=50, n_iterations=200, batch_size=25, learning_rate=0.001)
    rbm.fit(X)

    training_gen = rbm.training_recon

    # Plot images showing how the network progresses in getting better at
    # reconstructing the digits in the training set
    for epoch, batch in enumerate(training_gen):
        fig, axs = plt.subplots(5, 5)
        plt.suptitle("Restricted Boltzmann Machine")
        cnt = 0
        for i in range(5):
            for j in range(5):
                axs[i,j].imshow(batch[cnt].reshape((28, 28)), cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("rbm_%d.png" % epoch)
        plt.close()



if __name__ == "__main__":
    main()