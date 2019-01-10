import copy
import random
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Random Trainer #
class GeneticTrainer:
    def __init__(self, environment, newBot, scoringMethod, populationSize, generations, mutationRate=0.1, survivalRate=0.5):
        self.bots = []
        self.environment = environment
        self.newBot = newBot
        self.scoringMethod = scoringMethod
        self.populationSize = populationSize
        self.generations = generations
        self.mutationRate = mutationRate
        self.survivalRate = survivalRate

    def trainBots(self):
        self.bots = []
        for i in range(self.populationSize):
            self.bots.append(self.newBot())
        for g in range(self.generations):
            print('Generation', g, 'of', self.generations)
            self.runGeneration()
        return self

    def runGeneration(self):
        fitness = self.scoreBots()
        self.pruneBots(fitness)
        self.generateChildren()

    def getBestBot(self):
        fitness = self.scoreBots()
        maxFitness = max(fitness)
        for i, bot in enumerate(self.bots):
            if fitness[i] == maxFitness:
                return bot

    def pruneBots(self, fitness):
        while len(self.bots) > self.populationSize * self.survivalRate:
            selectedBotIndex = random.randrange(len(self.bots))
            survivalFitness = random.uniform(min(fitness), max(fitness))
            if fitness[selectedBotIndex] <= survivalFitness:
                del self.bots[selectedBotIndex]
                del fitness[selectedBotIndex]
                # del fitness to keep indexes matching

    def generateChildren(self):
        possibleParents = self.bots[:]
        while len(self.bots) < self.populationSize:
            if random.random() > 0.8: #20 % chance for new bot
                self.bots.append(self.newBot())
            else:
                firstParent = random.choice(possibleParents)
                secondParent = random.choice(possibleParents)
                child = copy.deepcopy(firstParent)
                for layerIndex in range(len(child.net.layers)):
                    firstParentWeights = firstParent.net.layers[layerIndex].weights
                    firstParentBias = firstParent.net.layers[layerIndex].bias
                    secondParentWeights = secondParent.net.layers[layerIndex].weights
                    secondParentBias = secondParent.net.layers[layerIndex].bias
                    for i in range(len(firstParentWeights)):
                        for j in range(len(firstParentWeights[i])):
                            weightChoice = random.choice([firstParentWeights[i][j], secondParentWeights[i][j]])
                            child.net.layers[layerIndex].weights[i][j] = weightChoice + (self.mutationRate * (2 * random.random() - 1))
                    for i in range(len(firstParentBias)):
                        biasChoice = random.choice([firstParentBias[i], secondParentBias[i]])
                        child.net.layers[layerIndex].bias[i] = biasChoice + (self.mutationRate * (2 * random.random() - 1))
                self.bots.append(child)

    def scoreBots(self, rounds=10):
        scores = []
        for i, bot in enumerate(self.bots):
            scores.insert(i, 0)
            for r in range(rounds):
                observation = self.environment.reset()
                for _ in range(10000):
                    observation, reward, done, info = self.environment.step(bot.action(observation))
                    score = self.scoringMethod(observation, reward)

                    # self.environment.render()
                    # print(observation, reward, done, info)
                    # print(score)
                    # input()

                    scores[i] += score
                    if done:
                        break

        m_score = max(scores)
        fitness = [x / m_score for x in scores]
        return fitness
