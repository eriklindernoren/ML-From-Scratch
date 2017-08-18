from __future__ import print_function
import sys
import os
import math
import random
import numpy as np
import progressbar
import gym

from sklearn.datasets import fetch_mldata

# Import helper functions
from mlfromscratch.utils.data_manipulation import to_categorical
from mlfromscratch.utils.optimizers import Adam
from mlfromscratch.utils.loss_functions import SquareLoss
from mlfromscratch.utils.layers import Dense, Dropout, Flatten, Activation, Reshape, BatchNormalization
from mlfromscratch.supervised_learning import NeuralNetwork

class DeepQLearning():
    def __init__(self, env_name='CartPole-v0', epsilon=1, gamma=0.9, decay_rate=0.005, min_epsilon=0.1):
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

        # Initialize the environment
        self.env = gym.make(env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
    
    def set_model(self, model):
        self.model = model(self.n_states, self.n_actions)

    def train(self, n_epochs=500, batch_size=32):
        max_reward = 0
        memory_limit = 500
        replay_history = []

        for epoch in range(n_epochs):
            state = self.env.reset()
            total_reward = 0

            epoch_loss = []
            while True:

                # Choose action randomly
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(self.n_actions)
                # Take action with highest predicted utility given state
                else:
                    action = np.argmax(self.model.predict(state), axis=1)[0]

                # Take a step
                new_state, reward, done, _ = self.env.step(action)
                replay_history.append([state, action, reward, new_state, done])

                # Sample batch from replay
                _batch_size = min(len(replay_history), batch_size)
                replay_batch = np.array(random.sample(replay_history, _batch_size))

                states = np.array([a[0] for a in replay_batch])
                new_states = np.array([a[3] for a in replay_batch])

                # Predict the expected utility of current state and new state
                Q = self.model.predict(states)
                Q_new = self.model.predict(new_states)

                X = np.empty((_batch_size, self.n_states))
                y = np.empty((_batch_size, self.n_actions))

                # Construct training data
                for i in range(_batch_size):
                    state_r, action_r, reward_r, new_state_r, done_r = replay_batch[i]
                    
                    target = Q[i]
                    target[action_r] = reward_r
                    # If we're done the utility is simply the accumulated reward,
                    # otherwise we add the expected maximum future reward as well
                    if not done_r:
                        target[action_r] += self.gamma * np.amax(Q_new[i])

                    X[i] = state_r
                    y[i] = target

                loss = self.model.train_on_batch(X, y)
                epoch_loss.append(loss)

                total_reward += reward
                state = new_state

                if done: break
            
            epoch_loss = np.mean(epoch_loss)
            # If memory limit is exceeded remove the oldest entry
            if len(replay_history) > memory_limit:
                replay_history.pop(0)

            # Reduce the epsilon parameter
            self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon) * np.exp(-self.decay_rate * epoch)
            
            max_reward = max(max_reward, total_reward)

            print ("%d [Loss: %.4f, Reward: %s, Epsilon: %.4f, Max Reward: %s]" % (epoch, epoch_loss, total_reward, self.epsilon, max_reward))

        print ("Training Finished")

    def play(self, n_epochs):
        # self.env = gym.wrappers.Monitor(self.env, '/tmp/cartpole-experiment-1', force=True)
        for epoch in range(n_epochs):
            state = self.env.reset()
            total_reward = 0
            while True:
                self.env.render()
                action = np.argmax(self.model.predict(state), axis=1)[0]
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done: break
            print ("%d Reward: %s" % (epoch, total_reward))


def main():
    dql = DeepQLearning()

    # Model builder
    def model(n_inputs, n_outputs):    
        clf = NeuralNetwork(optimizer=Adam(), loss=SquareLoss)
        clf.add(Dense(32, input_shape=(n_inputs,)))
        clf.add(Activation('relu'))
        clf.add(Dense(32))
        clf.add(Activation('relu'))
        clf.add(Dense(n_outputs))

        return clf

    dql.set_model(model)

    print ()
    dql.model.summary(name="Deep Q-Learning Model")

    dql.train(n_epochs=300)
    dql.play(n_epochs=100)

if __name__ == "__main__":
    main()