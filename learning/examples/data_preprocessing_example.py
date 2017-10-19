from Helper import DataPreprocessing

demo = DataPreprocessing()
demo.readSimpleDataSet("pca_dataset.txt", demo.ND_MAT, demo.FLOAT, "\t")
demo.graph2D(color="#9AB4FF")
result = demo.pca(1)
demo.graph2D()
demo.showGraph()