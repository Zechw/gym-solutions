import numpy as np
import tensorflow as tf
from tensorflow import keras
from baseGame import Actor, Game

class TensorActor(Actor):
    def __init__(self):
        self.game_history = []
        self.net = tf.keras.models.Sequential()
        self.net.add(keras.layers.Dense(3, input_shape=(5,), activation='sigmoid'))
        self.net.add(keras.layers.Dense(1))
        self.max_steps = 1

        self.net.compile(optimizer=tf.train.AdamOptimizer(0.001),
            loss='mse',
            metrics=['accuracy'])

    def action(self, observation):
        left = self.net.predict(np.array([np.append(observation,0)]))[0][0]
        right = self.net.predict(np.array([np.append(observation,1)]))[0][0]

        left += (np.random.random()-0.5) * 0.2 / self.max_steps
        right += (np.random.random()-0.5) * 0.2 / self.max_steps

        return 0 if left > right else 1

    def run(self):
        i = 0
        while True:
            i += 1
            print(' - ',i)
            game = Game()
            game.run(self)
            if i % 100 == 0:
                self.train()
                if len(self.game_history) > 1000:
                    self.game_history = []

    def train(self):
        data = []
        labels = []
        for game in self.game_history:
            steps_alive = len(game.rewards)
            self.max_steps = 1
            if steps_alive > self.max_steps:
                self.max_steps = steps_alive
            if steps_alive == 500 and i > 450:
                continue # max score, train on all but the end
            for i, observation in enumerate(game.observations):
                reward = min(1, (steps_alive - i)/steps_alive)

                data.append(np.append(observation, game.actions[i]))
                labels.append(reward)

        data = np.array(data)
        labels = np.array(labels)
        if data.size != 0:
            self.net.fit(data, labels, epochs=30)

    def on_game_end(self, game):
        self.game_history.append(game)


actor = TensorActor()
actor.run()
