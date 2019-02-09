import numpy as np
import tensorflow as tf
from tensorflow import keras
from baseGame import Actor, Game

class TensorActor(Actor):
    def __init__(self):
        self.round_batch_size = 100
        self.traning_epochs = 30
        self.max_memory_size = 500
        self.game_history = []
        self.build_net()

    def build_net(self):
        self.net = tf.keras.models.Sequential()
        self.net.add(keras.layers.Dense(3, input_shape=(5,), activation='sigmoid'))
        self.net.add(keras.layers.Dense(1))

        self.net.compile(optimizer=tf.train.AdamOptimizer(0.001),
            loss='mse',
            metrics=['accuracy'])

    def action(self, observation):
        high_score = max([len(g.rewards) for g in self.game_history[-100:]]) if len(self.game_history) > 0 else 0
        if np.random.random() > min(0.75, high_score / 500): #random factor up to solved value
            return 0 if np.random.random() > 0.5 else 1
        left = self.fire(observation, 0)
        right = self.fire(observation, 1)
        return 0 if left > right else 1

    def fire(self, observation, action):
        return self.net.predict(np.array([np.append(observation,action)]))[0][0]

    def run(self):
        i = 0
        while True:
            i += 1
            game = Game()
            game.run(self, False)
            avg = self.avg_last_n_games(100)
            print(i+1, "\t", len(game.rewards), "\t", avg)
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
            if max_steps > 490:
                continue
            for step_i, observation in enumerate(game.observations):
                reward = self.reward_function(game, game_i, step_i, max_steps)

                inputs.append(np.append(observation, game.actions[step_i]))
                desired_outpus.append(reward)

        inputs = np.array(inputs)
        desired_outpus = np.array(desired_outpus)
        if desired_outpus.size != 0:
            self.net.fit(inputs, desired_outpus, epochs=self.traning_epochs)

    def reward_function(self, game, game_i, step_i, max_steps):
        return (max_steps - step_i)/max_steps

    def avg_last_n_games(self, n):
        return sum([len(g.rewards) for g in self.game_history[-n:]])/n

    def on_game_end(self, game):
        self.game_history.append(game)
        if len(self.game_history) > self.max_memory_size:
            self.game_history.pop(0)

# actor = TensorActor()
# actor.run()
