import numpy as np
from functools import reduce

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, layerList):
        self.layers = []
        for i in range(len(layerList)-1):
            layer = NeuralLayer(layerList[i], layerList[i+1])
            self.layers.append(layer)

    def fire(self, inputs):
        return reduce(lambda input, layer: layer.fire(input), self.layers, inputs)

    def backPropagate(self, inputsList, desiredOutputsList, learningRate=0.05):
        currentOutputsList = []
        for inputs in inputsList:
            out = self.fire(inputs)
            currentOutputsList.append(out)
        currentOutputs = np.array(currentOutputsList)
        desiredOutputs = np.array(desiredOutputsList)
        loss = np.linalg.norm(currentOutputs-desiredOutputs)
        for q, layer in enumerate(reversed(self.layers)):
            i = len(self.layers) - 1 - q #obj layer index as we work backards
            #?

class NeuralLayer:
    def __init__(self, numInputs, numNeurons):
        self.weights = np.random.random((numInputs,numNeurons)) * 2 - 1
        self.bias = np.random.random(numNeurons) * 2 - 1

    def fire(self, inputs):
        return sigmoid(inputs @ self.weights + self.bias)
