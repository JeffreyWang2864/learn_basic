import numpy as np
import matplotlib.pyplot as plt
import os
from learning.Helper import Util

class Clustering:
    def __init__(self):
        self.DataSet = None
        self.ClusterValue = None
        self.Belongs = None
        self.ColorCode = [
            "#5C9EFF", "#FF5CE7", "#FFB15C", "#AEFF5C",
            "#695CFF", "#FF825C", "#5CFFE6", "#FFD65C",
            "#5CFF70", "#FFF85C", "#FF5C74", "#AD5CFF"
        ]
        self.Variation = 300
        self.Plot = plt.figure()
        self.__Figure = self.Plot.add_subplot(111)
    def readSimpleFile(self, path):
        assert isinstance(path, str)
        data = list()
        fr = open(Util().getDirectory() + "/DATA/" + path, 'r')
        lines = fr.readlines()
        for line in lines:
            tempLine = list()
            splitLine = line.strip().split('\t')
            tempLine = [float(item) for item in splitLine]
            data.append(tempLine.copy())
        self.DataSet = np.array(data)
        pass
    def __calcDistance(self, a, b):
        return np.sqrt(np.sum(np.power((a - b), 2)))
    def __randCenter(self, num):
        assert isinstance(num, int)
        horizontal = self.DataSet.shape[1]
        centroids = np.mat(np.zeros((num, horizontal)))
        for i in range(horizontal):
            minValue = np.min(self.DataSet[:, i])
            currentRange = float(np.max(self.DataSet[:, i]) - minValue)
            centroids[:, i] = minValue + currentRange * np.random.rand(num, 1)
        self.ClusterValue = centroids
    def kmeans(self, num):
        assert isinstance(num, int)
        vertical = self.DataSet.shape[0]
        clusterAssessment = np.mat(np.zeros((vertical, 2)))   #min index of each element, distance square of each data
        if self.ClusterValue is None:
            self.__randCenter(num)
        END_FLAG = True
        index = 0
        while END_FLAG:
            END_FLAG = False
            for i in range(vertical):
                minDistance = np.inf
                minIndex = -1
                for j in range(num):
                    currentDistance = self.__calcDistance(self.ClusterValue[j], self.DataSet[i])
                    if currentDistance < minDistance:
                        minDistance = currentDistance
                        minIndex = j
                if clusterAssessment[i, 0] != minIndex:
                    END_FLAG = True
                clusterAssessment[i] = minIndex, minDistance ** 2
            for i in range(num):
                pointsInCluster = self.DataSet[np.nonzero(np.array(clusterAssessment[:, 0]) == i)[0]]
                self.ClusterValue[i] = np.mean(pointsInCluster, axis=0)
                self.__Figure.scatter(self.ClusterValue[i, 0], self.ClusterValue[i, 1], marker='+',
                                      c=self.ColorCode[i], s=(500 + index * self.Variation), alpha=min(1, (index+1)/10))
            index += 1
        self.Belongs = np.array(clusterAssessment[:, 0].flatten().tolist()[0]).astype(int)
    def graph(self):
        for i in range(self.DataSet.shape[0]):
            self.__Figure.scatter(self.DataSet[i, 0], self.DataSet[i, 1], marker='o',
                                  c=self.ColorCode[self.Belongs[i]], s=100, alpha=0.2)
    def showPlot(self):
        plt.show()