from machinelearning import classify
from Helper import Util


model = classify.NaiveBayes()
model.ReadSimpleFile("sushiReview.txt")
test_data, test_label = model.SeparateDataSet(0.1)
model.CreateDictionary()
model.BuildModel()

model.ResultLabels = ("negative", "positive")

error, predicted = model.SmartTest(test_data, test_label)
print("error rate: %.6f%%"%(error*100))