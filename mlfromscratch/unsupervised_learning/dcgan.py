from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import progressbar
from sklearn.datasets import fetch_mldata

from mlfromscratch.deep_learning.optimizers import Adam
from mlfromscratch.deep_learning.loss_functions import CrossEntropy
from mlfromscratch.deep_learning.layers import Dense, Dropout, Flatten, Activation, Reshape, BatchNormalization, ZeroPadding2D, Conv2D, UpSampling2D
from mlfromscratch.deep_learning import NeuralNetwork


class DCGAN():
    def __init__(self):
        self.img_rows = 28 
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.channels, self.img_rows, self.img_cols)
        self.latent_dim = 100

        optimizer = Adam(learning_rate=0.0002, b1=0.5)
        loss_function = CrossEntropy

        # Build the discriminator
        self.discriminator = self.build_discriminator(optimizer, loss_function)

        # Build the generator
        self.generator = self.build_generator(optimizer, loss_function)

        # Build the combined model
        self.combined = NeuralNetwork(optimizer=optimizer, loss=loss_function)
        self.combined.layers.extend(self.generator.layers)
        self.combined.layers.extend(self.discriminator.layers)

        print ()
        self.generator.summary(name="Generator")
        self.discriminator.summary(name="Discriminator")

    def build_generator(self, optimizer, loss_function):
        
        model = NeuralNetwork(optimizer=optimizer, loss=loss_function)

        model.add(Dense(128 * 7 * 7, input_shape=(100,)))
        model.add(Activation('leaky_relu'))
        model.add(Reshape((128, 7, 7)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, filter_shape=(3,3), padding='same'))
        model.add(Activation("leaky_relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, filter_shape=(3,3), padding='same'))
        model.add(Activation("leaky_relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(1, filter_shape=(3,3), padding='same'))
        model.add(Activation("tanh"))

        return model

    def build_discriminator(self, optimizer, loss_function):
        
        model = NeuralNetwork(optimizer=optimizer, loss=loss_function)

        model.add(Conv2D(32, filter_shape=(3,3), stride=2, input_shape=self.img_shape, padding='same'))
        model.add(Activation('leaky_relu'))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, filter_shape=(3,3), stride=2, padding='same'))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(Activation('leaky_relu'))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, filter_shape=(3,3), stride=2, padding='same'))
        model.add(Activation('leaky_relu'))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, filter_shape=(3,3), stride=1, padding='same'))
        model.add(Activation('leaky_relu'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(2))
        model.add(Activation('softmax'))

        return model


    def train(self, epochs, batch_size=128, save_interval=50):

        mnist = fetch_mldata('MNIST original')

        X = mnist.data.reshape((-1,) + self.img_shape)
        y = mnist.target

        # Rescale -1 to 1
        X = (X.astype(np.float32) - 127.5) / 127.5

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.discriminator.set_trainable(True)

            # Select a random half batch of images
            idx = np.random.randint(0, X.shape[0], half_batch)
            imgs = X[idx]

            # Sample noise to use as generator input
            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of images
            gen_imgs = self.generator.predict(noise)

            valid = np.concatenate((np.ones((half_batch, 1)), np.zeros((half_batch, 1))), axis=1)
            fake = np.concatenate((np.zeros((half_batch, 1)), np.ones((half_batch, 1))), axis=1)

            # Train the discriminator
            d_loss_real, d_acc_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_acc = 0.5 * (d_acc_real + d_acc_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            # We only want to train the generator for the combined model
            self.discriminator.set_trainable(False)

            # Sample noise and use as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # The generator wants the discriminator to label the generated samples as valid
            valid = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))), axis=1)

            # Train the generator
            g_loss, g_acc = self.combined.train_on_batch(noise, valid)

            # Display the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, acc: %.2f%%]" % (epoch, d_loss, 100*d_acc, g_loss, 100*g_acc))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1 (from -1 to 1)
        gen_imgs = 0.5 * (gen_imgs + 1)

        fig, axs = plt.subplots(r, c)
        plt.suptitle("Deep Convolutional Generative Adversarial Network")
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,0,:,:], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=200000, batch_size=64, save_interval=50)


