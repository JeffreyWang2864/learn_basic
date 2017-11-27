import numpy as np
# import os
# os.chdir("/Users/Excited/Desktop")
#
# from machinelearning.regression import Regression
#
# demo = Regression()
#
# def loadDataSet(fileName):
#     dataMat = []
#     fr = open(fileName)
#     x, y = list(), list()
#     for line in fr.readlines():
#         curLine = line.strip().split('\t')
#         x.append(float(curLine[0]))
#         y.append(float(curLine[1]))
#     return np.array(np.mat(x).transpose()), np.array(y)
#     #return np.array(x), np.array(y)
#
# a, b = loadDataSet('/Users/Excited/PycharmProjects/learning/DATA/归档/sine.txt')
# demo.DataSet = a
# demo.Labels = b
# demo.Title = ['111', ]
# testD, testL = demo.SeparateDataSet()
#
# newD, newL = list(), list()
# aa = [float(item) for item in testD]
# sortIndex = np.array(aa).argsort()
# for item in sortIndex:
#     newD.append(float(testD[item]))
#     newL.append(float(testL[item]))
# newD = np.array(newD)
# newL = np.array(newL)
# newD = np.array(np.mat(newD).transpose())
#
# demo.MIN_NUM_DATA = 5
# demo.MIN_DELTA_ERR = 1.0
#
# demo.LinearRegression([0, ], mode=demo.TREE_BASED)
# rs, predicted = demo.SmartTest(newD, newL, weight=0.4, toggle_print=True)
# print(rs)
# demo.Graph(0, ('x', 'y'), [newD, predicted])


# demo.ReadSimpleFile("pca_dataset.txt")
# testD, testL = demo.SeparateDataSet()
# newD, newL = list(), list()
# aa = [float(item) for item in testL]
# sortIndex = np.array(aa).argsort()
# for item in sortIndex:
#     newD.append((testD[item]))
#     newL.append(float(testL[item]))
# newD = np.array(newD)
# newL = np.array(newL)
# demo.LinearRegression([0, ], mode=demo.RIDGE_REGRESSION, test_data=newD[:, [0, ]], test_label=newL)
# rs, predicted = demo.SmartTest(newD[:, [0, ]], newL)
# print(rs)
# demo.Graph(0, ('x', 'y'), [newD[:, [0, ]], predicted])

# demo.ReadSimpleFile("pca_dataset.txt")
# testD, testL = demo.SeparateDataSet()
# newD, newL = list(), list()
# aa = [float(item) for item in testL]
# sortIndex = np.array(aa).argsort()
# for item in sortIndex:
#     newD.append((testD[item]))
#     newL.append(float(testL[item]))
# newD = np.array(newD)
# newL = np.array(newL)
# demo.LinearRegression([0, ])
# rs, predicted = demo.SmartTest(newD[:, [0, ]], newL)
# print(rs)
# demo.Graph(0, ('x', 'y'), [newD[:, [0, ]], predicted])

# from Helper import DataPreprocessing
#
# demo = DataPreprocessing()
# demo.readParagraph("sushiReview.txt", True, '\t')
# demo.convertLevelToBool()
# demo.balanceDataSet()


# file = open("./dat.txt", 'r')
# lines = file.readline()
# rawData = list()
# for item in lines.split("\t"):
#     if len(item) > 1:
#         rawData.append(float(item))
#
# rawData = np.array(rawData)
# sortedIndexes = np.argsort(rawData)
# sortedData = np.sort(rawData)

# class UnionFind:
#     def __init__(self, length):
#         self.rank = np.array([1 for _ in range(length)])
#         self.parent = np.array([x for x in range(length)])
#         self.count = length
#     def find(self, target):
#         assert isinstance(target, int)
#         assert 0 <= target < self.count
#         while target != self.parent[target]:
#             target = self.parent[target]
#         return target
#     def isConnected(self, node1, node2):
#         return self.find(node1) == self.find(node2)
#     def unionElements(self, node1, node2):
#         root1 = self.find(node1)
#         root2 = self.find(node2)
#         if root1 == root2:
#             return
#         if self.rank[root1] < self.rank[root2]:
#             self.parent[root1] = root2
#         elif self.rank[root1] > self.rank[root2]:
#             self.parent[root2] = root1
#         else:
#             self.parent[root1] = root2
#             self.rank[root1] += 1
#
# if __name__ == '__main__':
#     demo = UnionFind(5)
#     demo.isConnected(1, 2)