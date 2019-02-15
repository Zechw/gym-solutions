import sys
import gym
import tensorflow as tf
from tensorflow import keras
from baseGame import GameFactory, Actor
from tensorNetwork import TensorActor
from qNetwork import QActor


gameFactory = GameFactory()

class TensorCartActor(TensorActor):
    def get_new_game(self):
        return gameFactory.get_env('CartPole-v1')

    def get_discrete_actions(self):
        return range(2)

    def build_net(self):
        self.net = tf.keras.models.Sequential()
        self.net.add(keras.layers.Dense(3, input_shape=(5,), activation='sigmoid'))
        self.net.add(keras.layers.Dense(1))

        self.net.compile(optimizer=tf.train.AdamOptimizer(0.001),
            loss='mse',
            metrics=['accuracy'])

    def get_best_action(self, observation):
        max_out = None
        max_action = None
        for action in self.get_discrete_actions():
            out = self.fire(observation, action)[0]
            if max_out is None or out > max_out:
                max_out = out
                max_action = action
        return max_action

    def score_game(self, game):
        return len(game.rewards)

    def reward_function(self, game, game_i, step_i, max_steps):
        return (max_steps - step_i)/max_steps

class QCartActor(QActor):
    def get_new_game(self):
        return gameFactory.get_env('CartPole-v1')

    def get_discrete_actions(self):
        return range(2)

    def build_net(self):
        self.net = tf.keras.models.Sequential()
        self.net.add(keras.layers.Dense(3, input_shape=(5,), activation='sigmoid'))
        self.net.add(keras.layers.Dense(1))

        self.net.compile(optimizer=tf.train.AdamOptimizer(0.001),
            loss='mse',
            metrics=['accuracy'])

    def score_game(self, game):
        return len(game.rewards)

    def current_reward(self, game, game_i, step_i, max_steps):
        return (max_steps - step_i)/max_steps

class QMountainActor(QActor):
    def get_new_game(self):
        return gameFactory.get_env('MountainCar-v0')

    def get_discrete_actions(self):
        return range(3)

    def build_net(self):
        self.net = tf.keras.models.Sequential()
        self.net.add(keras.layers.Dense(2, input_shape=(3,), activation='sigmoid'))
        self.net.add(keras.layers.Dense(1))

        self.net.compile(optimizer=tf.train.AdamOptimizer(0.001),
            loss='mse',
            metrics=['accuracy'])

    def current_reward(self, game, game_i, step_i, max_steps):
        return game.observations[step_i][0]


    def score_game(self, game):
        max_right = None
        for observation in game.observations:
            if max_right is None or observation[0] > max_right:
                max_right = observation[0]
        return max_right


modes = {
    'tensorcart' : TensorCartActor,
    'qcart' : QCartActor,
    'qmountain' : QMountainActor
}

args = sys.argv
try:
    env_arg = args[1]
except IndexError:
    env_arg = list(modes)[0]

actor_class = modes[env_arg]
print('running', actor_class)
actor = actor_class()
actor.run()
