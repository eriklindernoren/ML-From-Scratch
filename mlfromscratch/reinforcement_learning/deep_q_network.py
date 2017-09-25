from __future__ import print_function, division
import random
import numpy as np
import gym
from collections import deque


class DeepQNetwork():
    """Q-Learning with deep neural network to learn the control policy. 
    Uses a deep neural network model to predict the expected utility (Q-value) of executing an action in a given state. 

    Reference: https://arxiv.org/abs/1312.5602
    Parameters:
    -----------
    env_name: string
        The environment that the agent will explore. 
        Check: https://gym.openai.com/envs
    epsilon: float
        The epsilon-greedy value. The probability that the agent should select a random action instead of
        the action that will maximize the expected utility. 
    gamma: float
        Determines how much the agent should consider future rewards. 
    decay_rate: float
        The rate of decay for the epsilon value after each epoch.
    min_epsilon: float
        The value which epsilon will approach as the training progresses.
    """
    def __init__(self, env_name='CartPole-v1', epsilon=1, gamma=0.9, decay_rate=0.005, min_epsilon=0.1):
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.memory_size = 300
        self.memory = []

        # Initialize the environment
        self.env = gym.make(env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
    
    def set_model(self, model):
        self.model = model(n_inputs=self.n_states, n_outputs=self.n_actions)

    def _select_action(self, state):
        if np.random.rand() < self.epsilon:
            # Choose action randomly
            action = np.random.randint(self.n_actions)
        else:
            # Take action with highest predicted utility given state
            action = np.argmax(self.model.predict(state), axis=1)[0]

        return action

    def _memorize(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))
        # Make sure we restrict memory size to specified limit
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def _construct_training_set(self, replay):
        # Select states and new states from replay
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])

        # Predict the expected utility of current state and new state
        Q = self.model.predict(states)
        Q_new = self.model.predict(new_states)

        replay_size = len(replay)
        X = np.empty((replay_size, self.n_states))
        y = np.empty((replay_size, self.n_actions))
        
        # Construct training set
        for i in range(replay_size):
            state_r, action_r, reward_r, new_state_r, done_r = replay[i]

            target = Q[i]
            target[action_r] = reward_r
            # If we're done the utility is simply the reward of executing action a in
            # state s, otherwise we add the expected maximum future reward as well
            if not done_r:
                target[action_r] += self.gamma * np.amax(Q_new[i])

            X[i] = state_r
            y[i] = target

        return X, y

    def train(self, n_epochs=500, batch_size=32):
        max_reward = 0

        for epoch in range(n_epochs):
            state = self.env.reset()
            total_reward = 0

            epoch_loss = []
            while True:

                action = self._select_action(state)
                # Take a step
                new_state, reward, done, _ = self.env.step(action)

                self._memorize(state, action, reward, new_state, done)

                # Sample replay batch from memory
                _batch_size = min(len(self.memory), batch_size)
                replay = random.sample(self.memory, _batch_size)

                # Construct training set from replay
                X, y = self._construct_training_set(replay)

                # Learn control policy
                loss = self.model.train_on_batch(X, y)
                epoch_loss.append(loss)

                total_reward += reward
                state = new_state

                if done: break
            
            epoch_loss = np.mean(epoch_loss)

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
