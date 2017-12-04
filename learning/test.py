from learning.Helper import DataPreprocessing

demo = DataPreprocessing()

demo.readSimpleDataSet("testSet.txt", demo.SETTYPE_NDARRAY, demo.DATATYPE_FLOAT, add_label=True)
demo.writeDataSet("export", demo.FILE_XML, True)





demo.readXML("export", set_form=demo.SETTYPE_NDARRAY, data_form=demo.DATATYPE_FLOAT, add_label=True)
print(demo.DataSet.shape, len(demo.Label))