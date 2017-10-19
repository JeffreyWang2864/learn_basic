from machinelearning.classify import DecisionTree

demo = DecisionTree()
demo.ReadSimpleFile("decis.txt")
test_data = demo.SeparateDataSet()
demo.BuildTree()
demo.VisualizeTree()
error, predict_data = demo.SmartTest(test_data)
print(error)