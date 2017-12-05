# from learning.Helper import DataPreprocessing
#
# demo = DataPreprocessing()
#
# demo.readSimpleDataSet("abalone.txt", demo.SETTYPE_NDARRAY, demo.DATATYPE_FLOAT, add_label=True)
# demo.writeDataSet("abalone", demo.FILE_XML, True)
#
#
# demo.readXML("abalone", set_form=demo.SETTYPE_NDARRAY, data_form=demo.DATATYPE_FLOAT, add_label=True)
# print(demo.DataSet.shape, len(demo.Label))

import random
from matplotlib import pyplot as plt

def startSimulation(epoch):
     assert isinstance(epoch, int)
     assert 0 < epoch
     greaterThan = int()
     for _ in range(epoch):
         targetList = [i for i in range(1, 16)]
         selections = list()
         rangeNumber = len(targetList) - 1
         for _ in range(8):
             selections.append(targetList.pop(random.randint(0, rangeNumber)))
             rangeNumber -= 1
         comparingNumber = targetList.pop(random.randint(0, rangeNumber))
         for selection in selections:
             if comparingNumber > selection:
                 greaterThan += 1
                 break
     return ((epoch - greaterThan) / float(epoch) * 100)

if __name__ == '__main__':
    epoch = 100

    epochs = list()
    results = list()

    for i in range(20):
        result = startSimulation(epoch)
        print("epoch: %d\t\tresult: %.3f%%"%(epoch, result))
        epochs.append(epoch)
        results.append(result)
        epoch += int(epoch/2)

    print("average result: %.3f%%"%(sum(results)/len(results)))

    plt.xlabel("epoch")
    plt.ylabel("rate of smaller")
    plt.ylim(0, 100)
    plt.scatter(epochs, results)
    plt.plot(epochs, results)
    plt.show()