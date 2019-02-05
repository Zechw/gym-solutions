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
        self.model.add(keras.layers.Dense(4))
        self.model.add(keras.layers.Dense(2, activation='softmax'))

        self.model.compile(optimizer=tf.train.AdamOptimizer(0.001),
            loss='mse',
            metrics=['accuracy'])

        # q-net to predict score based on inputs, action
        self.q = tf.keras.models.Sequential()
        self.q.add(keras.layers.Dense(5))
        self.q.add(keras.layers.Dense(6, activation='sigmoid'))
        self.q.add(keras.layers.Dense(1))

        self.q.compile(optimizer=tf.train.AdamOptimizer(0.001),
            loss='mse',
            metrics=['accuracy'])


    def action(self, observation):
        # result = self.model.predict(np.array([observation]))
        # # print(result)
        # action =  result[0].argmax()
        #
        #  just take best binary q
        left = self.q.predict(np.array([np.append(observation,0)]))[0][0]
        right = self.q.predict(np.array([np.append(observation,1)]))[0][0]

        left += np.random.random() * .1
        right += np.random.random() * .1

        # print(left, right)
        return 0 if left > right else 1

    def learn(self, histories):
        print('--learning--')
        # print(histories)
        data = []
        labels = []
        for game in histories:
            final_reward = game[1]
            for i, state in enumerate(game[0]):
                desired_q = (final_reward - i) / final_reward
                data.append(np.append(state[0], state[1]))
                labels.append(desired_q)

        data = np.array(data)
        labels = np.array(labels)
        self.q.fit(data, labels, epochs=30)

        # input()


def render(bot, env):
    observation = env.reset()
    i = 0
    game_i = 0
    current_match_history = [] # list of (observation, action)    ##  , reward)
    all_matches_history = [] # list of resulting (current_match_history, reward)
    while True:
    #for _ in range(10000):
        i += 1
        action = bot.action(observation)
        current_match_history.append((observation, action))
        observation, reward, done, info = env.step(action)
        env.render()
        # time.sleep(0.01)

        # print(observation, reward, done, info)
        # input()

        if done:
            game_i += 1
            print(i)
            # print(current_match_history)
            all_matches_history.append((current_match_history, i))
            if game_i % 100 == 0: #train every n games
                bot.learn(all_matches_history)
                all_matches_history = []

            # print(current_match_history)

            current_match_history = []

            i = 0
            # time.sleep(0.1)
            observation = env.reset()




render(Actor(), gym.make('CartPole-v1'))
