from learning.machinelearning.regression import Regression
from learning.Helper import DataPreprocessing
import numpy as np

data = DataPreprocessing()
data.readSimpleDataSet("house_price.txt", set_form=data.SETTYPE_NDARRAY,
                       data_form=data.DATATYPE_FLOAT, sep=" ", add_title=True, add_label=True)
title = data.ExtraData
rawData = data.DataSet
label = data.Label

model = Regression()
model.DataSet = rawData
model.Labels = label
model.Title = title

testData, testLabel = model.separateDataSet()

sortedIndex = testLabel.argsort()
newD = np.vstack([testData[i] for i in sortedIndex])
newL = np.array([testLabel[i] for i in sortedIndex])

model.linearRegression(mode="DEFAULT")
r_square, predictedData = model.smartTest(newD, newL, toggle_print=True, weight=2.5)

print("R Square: %.6f"%r_square)
model.graph(0, labels=('x', 'y'), line=[newD[:, [0, ]], predictedData])