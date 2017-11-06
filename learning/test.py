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

a, b = loadDataSet('/Users/Excited/PycharmProjects/learning/DATA/归档/exp2.txt')
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

demo.LinearRegression([0, ], mode=demo.TREE_BASED)
rs, predicted = demo.SmartTest(newD, newL, weight=0.4, toggle_print=True)
print(rs)
demo.Graph(0, ('x', 'y'), [newD, predicted])


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
# dictionary, dataMat = demo.wordBagging(demo.ENGLISH, demo.ND_ARRAY, lambda x: len(x) > 3)
# demo.DataSet = dataMat
# demo.removeRedundantData()
