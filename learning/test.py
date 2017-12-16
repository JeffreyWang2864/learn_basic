from learning.Helper import DataPreprocessing

demo = DataPreprocessing()
demo.readSimpleDataSet("association_test.txt", demo.SETTYPE_LIST, demo.DATATYPE_INT, add_label=False, sep=" ")
demo.writeDataSet("association_test", demo.FILE_XML, False)