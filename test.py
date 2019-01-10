import gym
import time
import random
from neuralNetwork import sigmoid

from neuralNetwork import NeuralNetwork
from trainer import GeneticTrainer

import numpy as np

# Game #
class Actor:
    def __init__(self, netMap=[4,2]):
        self.net = NeuralNetwork(netMap)

    def action(self, observation):
        pass ## IMPLEMENT ME! ##

    def learn(self, observation, action, reward):
        pass

def render(bot, env):
    observation = env.reset()
    i = 0
    for _ in range(10000):
        i += 1
        observation, reward, done, info = env.step(bot.action(observation))
        env.render()
        time.sleep(0.01)

        print(observation, reward, done, info)
        # input()

        if done:
            print(i)
            i = 0
            time.sleep(0.1)
            observation = env.reset()

class BoxActor(Actor):
    def action(self, observation):
        return self.net.fire(observation)

class PenActor(BoxActor):
    def action(self, observation):
        return (super(PenActor, self).action(observation) * 2) - 2

class DiscreteActor(Actor):
    def action(self, observation):
        result = self.net.fire(observation)
        return random.choice([i for i, x in enumerate(result) if x == max(result)])

## poleGame ##
# e = gym.make('CartPole-v1')
# o = e.reset()
# a = lambda: DiscreteActor([4,2])
# s = lambda o, r: r - sigmoid(abs(o[0]) + abs(o[2]))
# b = GeneticTrainer(e, a, s, 10, 30).trainBots().getBestBot()
# render(b, e)

## mountainCar ##
# e = gym.make('MountainCarContinuous-v0')
# o = e.reset()
# a = lambda: BoxActor([2,4,1])
# # s = lambda o, r: 0.2 * sigmoid(o[0]) + sigmoid(abs(o[1])) # value right side. could also value velocity
# s = lambda o, r: sigmoid(r)
# b = GeneticTrainer(e, a, s, 20, 30).trainBots().getBestBot()
# render(b, e)

## pendulum ##
e = gym.make('Pendulum-v0')
o = e.reset()
a = lambda: BoxActor([3,1])
s = lambda o, r: sigmoid(r)
b = GeneticTrainer(e, a, s, 5, 30).trainBots().getBestBot()
render(b, e)

## acro ##
# e = gym.make('Acrobot-v1')
# o = e.reset()
# a = lambda: DiscreteActor([6,3])
# s = lambda o, r: sum(o)
# b = GeneticTrainer(e, a, s, 5, 5).trainBots().getBestBot()
# render(b, e)
