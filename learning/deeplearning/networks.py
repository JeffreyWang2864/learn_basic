import numpy as np
from learning.machinelearning import classify as mc
from learning.machinelearning import regression as mr

class BackPropagation:
    def __init__(self, size):
        self.layers = len(size)
        self.size = size
        self.biases = [np.random.randn(num, 1) for num in size[1::]]
        self.weights = [np.random.randn(curNum, preNum) for curNum, preNum in zip(size[::-1], size[1::])]
        self.EPOCHS = 0
    def GD(self, trainData, eta):    #前向传播
        for i in range(self.EPOCHS):
            np.random.shuffle(trainData)
            nabla_bias = [np.zeros(b.shape) for b in self.biases]
            nabla_weight = [np.zeros(w.shape) for w in self.weights]
            for x, y in trainData:
                delta_nabla_b, delta_nabla_w = self.update(x, y)
                nabla_bias = [nb + dnb for nb, dnb in zip(nabla_bias, delta_nabla_b)]
                nabla_weight = [nw + dnw for nw, dnw in zip(nabla_weight, delta_nabla_w)]
            self.weights = [w - (eta) * nw for w, nw in zip(self.weights, nabla_bias)]
            self.biases = [b - (eta) * nb for b, nb in zip(self.biases, nabla_weight)]
            print("epoch %d completed"%i)

    def update(self, x, y):
        nabla_bias = [np.zeros(b.shape) for b in self.biases]
        nabla_weight = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        results = list()
        for bias, weight in zip(self.biases, self.weights):
            result = np.dot(weight, activation) + bias
            results.append(result)
            activations.append(self.sigmoid(result))
        #反向更新



    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))