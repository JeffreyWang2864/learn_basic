import numpy as np

from learning.Helper import Util
from learning.machinelearning.classify import SupportVectorMachine as svm

demo = svm()
demo.ReadSimpleFile("testSet.txt")          #read data from file
test_data, test_label = demo.SeparateDataSet()      #get test data and test label
#demo.RadicalBias_Gaussian(20, 0.01, 200, "DISABLED")
demo.Platt_SMO(20, 0.01, 200, "DISABLED")       #linear classification
demo.GetLine(True)              #calculate the  decision boundary
predicted = list()          #store the predict values
for i in range(len(test_data)):
    res = demo.Predict(np.array(test_data[i]).transpose(), test_label[i])   #predict
    print(res)
    predicted.append(res[1])        #append the predicted value
demo.GraphPoints()              #graph
Util().plot_ROC_curve(test_label, predicted)      #graph roc curve