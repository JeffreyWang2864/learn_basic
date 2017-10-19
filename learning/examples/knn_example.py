from machinelearning.classify import knn

demo = knn()
demo.ReadSimpleFile("datingTestSet.txt")
demo.Normolization()
testset, testdata = demo.SeparateDataSet()
result = demo.SmartTest(testset, testdata, toggle_print=True)
print(result)
demo.graph([0, 1])