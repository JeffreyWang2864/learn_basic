from learning.Helper import DataPreprocessing

demo = DataPreprocessing()
# demo.readSimpleDataSet("pca_dataset.txt", demo.ND_ARRAY, demo.FLOAT)
#
# demo.writeDataSet("export", demo.FILE_XML, False)

demo.readXML("export")