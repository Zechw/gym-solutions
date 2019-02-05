import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym

## todo clean up copy-pasta, factor into exportable class

# Game #
class Actor:
    def __init__(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(keras.layers.Dense(4, activation='relu'))
        self.model.add(keras.layers.Dense(2, activation='softmax'))

        self.model.compile(optimizer=tf.train.AdamOptimizer(0.001),
            loss='mse',
            metrics=['accuracy'])

    def action(self, observation):
        result = self.model.predict(np.array([observation]))
        print(result)
        return result[0].argmax()

    def learn(self, history, reward):
        pass

def render(bot, env):
    observation = env.reset()
    i = 0
    current_match_history = [] # list of (observation, action)  ## or ? ..., reward)
    all_matches_history = [] # list of resulting (current_match_history, reward)
    for _ in range(10000):
        i += 1
        action = bot.action(observation)
        current_match_history.append((observation, action))
        observation, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.01)

        # print(observation, reward, done, info)
        # input()

        if done:
            print(i)
            print(current_match_history)
            all_matches_history.append(current_match_history)
            bot.learn(current_match_history, i)
            current_match_history = []

            i = 0
            time.sleep(0.1)
            observation = env.reset()




render(Actor(), gym.make('CartPole-v1'))
