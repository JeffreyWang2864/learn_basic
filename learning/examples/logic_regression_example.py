from learning.machinelearning.classify import LogicRegression

demo = LogicRegression()
demo.ReadSimpleFile("testSet2.txt")
testSet, testLabel = demo.SeparateDataSet()

demo.GradientAscent()

#demo.StochasticGradientAscent()

line = demo.GetLine()
demo.Graph(line)
demo.SmartTest(testSet, testLabel)