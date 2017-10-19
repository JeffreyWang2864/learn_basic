import numpy as np
import machinelearning.classify as mc

class BackPropagation:
    def __init__(self, size):
        self.layers = len(size)
        self.size = size
        self.bias = [np.random.randn(num, 1) for num in size[1::]]
        self.wights = [np.random.randn(curNum, preNum) for curNum, preNum in zip(size[1::], size[::-1])]