from machinelearning.clustering import Clustering

demo = Clustering()
demo.readSimpleFile("kmean_testSet2.txt")
demo.kmeans(3)
demo.graph()
demo.showPlot()