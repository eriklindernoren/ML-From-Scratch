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

    # Training error plot
    training, = plt.plot(range(len(rbm.training_errors)), rbm.training_errors, label="Training Error")
    plt.legend(handles=[training])
    plt.title("Error Plot")
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()

    # Get the images that were reconstructed during training
    gen_imgs = rbm.training_reconstructions

    # Plot the reconstructed images during the first iteration
    fig, axs = plt.subplots(5, 5)
    plt.suptitle("Restricted Boltzmann Machine - First Iteration")
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i,j].imshow(gen_imgs[0][cnt].reshape((28, 28)), cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("rbm_first.png")
    plt.close()

    # Plot the images during the last iteration
    fig, axs = plt.subplots(5, 5)
    plt.suptitle("Restricted Boltzmann Machine - Last Iteration")
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i,j].imshow(gen_imgs[-1][cnt].reshape((28, 28)), cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("rbm_last.png")
    plt.close()


if __name__ == "__main__":
    main()