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
    def __init__(self):
        super(QMountainActor, self).__init__()
        self.round_batch_size = 50
        self.traning_epochs = 15
        self.max_possible_score = 1

    def get_new_game(self):
        return gameFactory.get_env('MountainCar-v0')

    def get_discrete_actions(self):
        return range(3)

    def build_net(self):
        self.net = tf.keras.models.Sequential()
        self.net.add(keras.layers.Dense(4, input_shape=(3,), activation='relu'))
        self.net.add(keras.layers.Dense(2, activation='sigmoid'))
        self.net.add(keras.layers.Dense(1))

        self.net.compile(optimizer=tf.train.AdamOptimizer(0.001),
            loss='mse',
            metrics=['accuracy'])

    def current_reward(self, game, game_i, step_i, max_steps):
        return 1 if step_i > 0 and game.observations[step_i][0] > max([o[0] for o in game.observations[:step_i]]) else 0
        #only give reward if new rightmost score

        # is (not first move and) farther right than last step
        # return 1 if step_i > 0 and game.observations[step_i][0] > game.observations[step_i-1][0] else 0

        #sum of +x location so far # return sum([1+o[0] for o in game.observations[:step_i+1]])
        # rightness # game.observations[step_i][0]
        # return 1 + game.observations[step_i][0]

    def score_game(self, game):
        return max([1+o[0] for o in game.observations])

class DebugMountainActor(Actor):
    def run(self, _):
        while True:
            game = gameFactory.get_env('MountainCar-v0')
            game.run(self, True)

    def action(self, observation):
        print(observation)
        inp = input()
        return 0 if inp == '' else int(inp)

actors = [
    TensorCartActor,
    QCartActor,
    QMountainActor,
    DebugMountainActor
]
print('Available Actors:')
for i, actor in enumerate(actors):
    print(' ', i, actor)

mode = input('which one would you like to run? ')
render_inp = input('would you like to render? y/n ')
render = True if render_inp == 'y' or render_inp == '' else False
actor_class = actors[int(mode)]
print('running', actor_class)
actor = actor_class()
actor.run(render)
