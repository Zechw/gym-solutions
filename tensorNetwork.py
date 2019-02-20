import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from baseGame import Actor, Game

class TensorActor(Actor):
    def __init__(self):
        self.round_batch_size = 50
        self.traning_epochs = 50
        self.max_memory_size = 500
        self.max_possible_score = 500
        self.high_score_reflection_window = 100
        self.game_history = []
        self.build_net()

    def build_net(self):
        raise Exception('IMPLEMENT')

    def get_discrete_actions(self):
        raise Exception('IMPLEMENT')

    def get_best_action(self, observation):
        raise Exception('IMPLEMENT')

    def reward_function(self, game, game_i, step_i, max_steps):
        raise Exception('IMPLEMENT')

    def get_new_game(self):
        raise Exception('IMPLEMENT')

    def score_game(self, game):
        raise Exception('IMPLEMENT')

    # main gym method
    def action(self, observation):
        high_score = self.high_score()
        if np.random.random() > min(0.75, high_score / self.max_possible_score): #random factor up to solved value
            return random.choice(self.get_discrete_actions())

        best_action = self.get_best_action(observation)
        return best_action

    def fire(self, observation, action):
        return self.net.predict(np.array([np.append(observation,action)]))[0]


    # main runner
    def run(self, render=True):
        i = 0
        while True:
            i += 1
            game = self.get_new_game()
            game.run(self, render)
            avg = self.avg_last_n_games(self.high_score_reflection_window)
            print(i+1, "\t", self.score_game(game), "\t", avg)
            del game
            # if avg >= 195 and len(self.game_history) >= 100:
            #     raise Exception('You win!')
            if i % self.round_batch_size == 0:
                self.train()

    def train(self):
        inputs = []
        desired_outpus = []
        print('--training--')
        for game_i, game in enumerate(self.game_history):
            max_steps = len(game.rewards)
            # if max_steps > self.max_possible_score - 20:
            #     # TODO make this off max frames, not score
            #     continue
            for step_i, observation in enumerate(game.observations):
                reward = self.reward_function(game, game_i, step_i, max_steps)

                inputs.append(np.append(observation, game.actions[step_i]))
                desired_outpus.append(reward)

        inputs = np.array(inputs)
        desired_outpus = np.array(desired_outpus)
        if desired_outpus.size != 0:
            self.net.fit(inputs, desired_outpus, epochs=self.traning_epochs)

    def high_score(self):
        # TODO caching?
        window = -1 * self.high_score_reflection_window
        return  max([self.score_game(g) for g in self.game_history[window:]]) if len(self.game_history) > 0 else 0

    def avg_last_n_games(self, n):
        return sum([self.score_game(g) for g in self.game_history[-n:]])/n

    def on_game_end(self, game):
        self.game_history.append(game)
        if len(self.game_history) > self.max_memory_size:
            self.game_history.pop(0)
