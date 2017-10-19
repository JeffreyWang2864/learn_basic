import numpy as np

from machinelearning.regression import Regression

demo = Regression()

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    x, y = list(), list()
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        x.append(float(curLine[0]))
        y.append(float(curLine[1]))
    return np.array(np.mat(x).transpose()), np.array(y)
    #return np.array(x), np.array(y)

a, b = loadDataSet('/Users/Excited/PycharmProjects/learning/DATA/归档/sine.txt')
demo.DataSet = a
demo.Labels = b
demo.Title = ['111', ]
testD, testL = demo.SeparateDataSet()

newD, newL = list(), list()
aa = [float(item) for item in testD]
sortIndex = np.array(aa).argsort()
for item in sortIndex:
    newD.append(float(testD[item]))
    newL.append(float(testL[item]))
newD = np.array(newD)
newL = np.array(newL)
newD = np.array(np.mat(newD).transpose())

demo.MIN_NUM_DATA = 5
demo.MIN_DELTA_ERR = 1.0

demo.LinearRegression([0, ], mode=demo.RIDGE_REGRESSION)
rs, predicted = demo.SmartTest(newD, newL, weight=0.4, toggle_print=True)
print(rs)
demo.Graph(0, ('x', 'y'), [newD, predicted])


# demo.ReadSimpleFile("abalone.txt")
# testD, testL = demo.SeparateDataSet()
# newD, newL = list(), list()
# aa = [float(item) for item in testL]
# sortIndex = np.array(aa).argsort()
# for item in sortIndex:
#     newD.append((testD[item]))
#     newL.append(float(testL[item]))
# newD = np.array(newD)
# newL = np.array(newL)
# demo.LinearRegression([1, 2, 3, 4, 5, 6, 7], mode=demo.TREE_BASED)
# rs, predicted = demo.SmartTest(newD[:, 1:8], newL)
# print(rs)
# demo.Graph(4, ('x', 'y'), [testD[:, [4, ]], predicted])

# demo = Clustering()
# demo.readSimpleFile("kmean_testSet2.txt")
# demo.kmeans(3)
# demo.graph()
# demo.showPlot()

# demo = knn()
# demo.ReadSimpleFile("datingTestSet.txt")
# demo.Normolization()
# testset, testdata = demo.SeparateDataSet()
# result = demo.SmartTest(testset, testdata, toggle_print=True)
# print(result)