import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
import pdb

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
        self.q.add(keras.layers.Dense(1, activation='sigmoid'))

        self.q.compile(optimizer=tf.train.AdamOptimizer(0.0001),
            loss='mse',
            metrics=['accuracy'])

        self.random_offset = 1


    def action(self, observation):
        # result = self.model.predict(np.array([observation]))
        # # print(result)
        # action =  result[0].argmax()
        #
        #  just take best binary q
        left = self.q.predict(np.array([np.append(observation,0)]))[0][0]
        right = self.q.predict(np.array([np.append(observation,1)]))[0][0]

        left += (np.random.random() - 0.5) * self.random_offset
        right += (np.random.random() - 0.5) * self.random_offset

        # print(left, right)
        return 0 if left > right else 1

    def secondsToLive(self, state):

        left_action = np.append(state,0)
        left_action = np.expand_dims(left_action,axis=0)

        right_action = left_action.copy()
        right_action[:,4] = 1

        left = self.q.predict( left_action )[0][0]
        right = self.q.predict(right_action)[0][0]

        return max(left,right)

    def learn(self, histories):
        self.random_offset *= 0.5
        print('--learning--')
        # print(histories)
        data = []
        labels = []
        for game in histories:
            survive_time = game[1]
            state_actions = game[0]
            won = survive_time == 500
            look_ahead_time = 20

            if survive_time > 500-look_ahead_time:
                state_actions = state_actions[-20:]

            print('debug')
            print(survive_time)


            for i, state_action in enumerate(state_actions):

                state = state_action[0]
                action = state_action[1]

                print(self.secondsToLive(state))
                if i < survive_time-1:
                    step_reward = 1

                    step_reward += look_ahead_time*self.secondsToLive(state_actions[i+1][0])
                else:
                    if won:
                        step_reward = 1
                        step_reward += look_ahead_time*self.secondsToLive(state_actions[i+1][0])
                    else:
                        step_reward = 0

                data.append(np.append(state[0], state[1]))
                labels.append(step_reward)

            print(labels)
        data = np.array(data)
        labels = np.array(labels)
        self.q.fit(data, labels, epochs=50)

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
            print(game_i, i)
            # print(current_match_history)
            all_matches_history.append((current_match_history, i))
            if game_i % 10 == 0: #train every n games
                bot.learn(all_matches_history)
                all_matches_history = []

            current_match_history = []

            i = 0
            # time.sleep(0.1)
            observation = env.reset()




render(Actor(), gym.make('CartPole-v1'))
