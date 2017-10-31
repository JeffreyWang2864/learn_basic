from machinelearning.regression import Regression
from Helper import DataPreprocessing
import numpy as np

data = DataPreprocessing()
data.readSimpleDataSet("house_price.txt", set_form=data.ND_ARRAY,
                       data_form=data.FLOAT, sep=" ", add_title=True, add_label=True)
title = data.ExtraData
rawData = data.DataSet
label = data.Label

model = Regression()
model.DataSet = rawData
model.Labels = label
model.Title = title

testData, testLabel = model.SeparateDataSet()

sortedIndex = testLabel.argsort()
newD = np.vstack([testData[i] for i in sortedIndex])
newL = np.array([testLabel[i] for i in sortedIndex])

model.LinearRegression(mode="DEFAULT")
r_square, predictedData = model.SmartTest(newD, newL, toggle_print=False, weight=2.5)

print("R Square: %.6f"%r_square)
model.Graph(0, labels=('x', 'y'), line=[newD[:, [0, ]], predictedData])