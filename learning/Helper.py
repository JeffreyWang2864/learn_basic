import os
import numpy as np
from matplotlib import pyplot as plt
import re
import jieba

class Util:
    def __init__(self):
        self.SeparateDataSet_Pattern = None
    def SelectRandomItem(self, current, target):
        if isinstance(current, int):
            new = current
            while new == current:
                new = int(np.random.uniform(0, target))
            return new
        elif isinstance(current, list or tuple):
            new = int(np.random.uniform(0, target))
            while new in current:
                new = int(np.random.uniform(0, target))
            return new
    def ClipStepSize(self, max, target, min):
        if target > max:
            return max
        if min > target:
            return min
        return target
    def EUCLID_SIM(self, d1, d2):
        return 1.0 / (1.0 + np.linalg.norm(d1 - d2))
    def PEARSON_SIM(self, d1, d2):
        if len(d1) < 3: return 1.0
        return 0.5 + 0.5 * np.corrcoef(d1, d2, rowvar=0)[0][1]
    def COSINE_SIM(self, d1, d2):
        num = float(d1.transpose() * d2)
        denom = np.linalg.norm(d1) * np.linalg.norm(d2)
        return 0.5 + 0.5 * (num / denom)
    def SplitDataSet(self, DataLen, test_proportion = 0.2, mode ="DEFAULT"):
        """
        mode --- decides the way function operates with your input data
            DEFAULT --- just split the data randomly
            LOAD    --- use saved split pattern
            SAVE    --- save the split pattern after separating your input, than return the table
        """
        if mode == "LOAD":
            assert self.SeparateDataSet_Pattern != None
            return self.SeparateDataSet_Pattern
        else:
            assert isinstance(DataLen, int)
            length_test = int(DataLen*test_proportion)
            ExistedElements = list()
            Tabel = [0] * DataLen
            for i in range(length_test):
                index = self.SelectRandomItem(ExistedElements, DataLen)
                Tabel[index] += 1
                ExistedElements.append(index)
            if mode == "SAVE":
                self.SeparateDataSet_Pattern = Tabel
            return Tabel
    def GetDirectory(self):
        return os.path.dirname(os.path.abspath(__file__))
    def plot_ROC_curve(self, DataLabel, PredictLabel):
        if isinstance(DataLabel, list):
            DataLabel = np.array(DataLabel)
        if isinstance(PredictLabel, list):
            PredictLabel = np.array(PredictLabel)
        StepIndex = DataLabel.argsort()
        cursor = (1.0, 1.0)
        HorizontalSum = float()
        PositiveNum = sum(PredictLabel == 1)
        y_step = 1 / float(PositiveNum)
        x_step = 1 / float(len(PredictLabel) - PositiveNum)
        graph = plt.figure()
        graph.clf()
        figure = plt.subplot(111)
        for index in StepIndex.tolist():
            if PredictLabel[index] == 1.0 or PredictLabel[index] == 1:
                x_var, y_var = 0, y_step
            else:
                x_var, y_var = x_step, 0
                HorizontalSum += cursor[1]
            figure.plot([cursor[0], cursor[0] - x_var], [cursor[1], cursor[1] - y_var], c='#5C9EFF')
            cursor = (cursor[0] - x_var, cursor[1] - y_var)
        figure.plot([0, 1], [0, 1], c='#A9CCFF', ls="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.axis([0, 1, 0, 1])
        print("the area covers %.2f%% of area" % ((HorizontalSum * x_step) * 100))
        plt.show()

class DataPreprocessing:
    def __init__(self):
        self.LIST = list
        self.ND_ARRAY = np.array
        self.ND_MAT = np.mat
        self.__SET_FORMAT = (self.LIST, self.ND_ARRAY, self.ND_MAT)
        self.INT = int
        self.FLOAT = float
        self.STR = str
        self.__DATA_FORMAT = (self.INT, self.FLOAT, self.STR)
        self.CHINESE = re.compile(r'([\u4E00-\u9FA5]+|\w+)')
        self.ENGLISH = re.compile(r'[a-z|A-Z]+')
        self.__LANGUAGE = (self.CHINESE, self.ENGLISH)
        self.DataSet = None
        self.Label = None
        self.graph = None
        self.ExtraData = None
    def __initGraph(self):
        self.graph = plt.figure()
    def __validPath(self, path):
        assert isinstance(path, str)
        if path[-4::] != ".txt":
            raise TypeError("Read file only support .txt format!")
        if not os.path.exists(Util().GetDirectory() + "/DATA/" + path):
            print("File does not exist: %s" % path)
            return False
        return True
    def readSimpleDataSet(self, path, set_form, data_form, sep ="\t", add_title = False, add_label = False):
        assert set_form in self.__SET_FORMAT
        assert data_form in self.__DATA_FORMAT
        assert isinstance(add_title, bool)
        assert isinstance(add_label, bool)
        assert self.__validPath(path)
        file = open(Util().GetDirectory() + "/DATA/" + path, "r")
        data = list()
        lines = file.readlines()
        if add_label:
            self.Label = list()
        if add_title:
            titleLine = lines.pop(0)
            self.ExtraData = titleLine.strip().split(" ")
        for line in lines:
            tempData = list()
            splitData = line.strip().split(sep)
            tempData = [data_form(item) for item in splitData]
            if add_label:
                self.Label.append(tempData.pop())
            data.append(tempData.copy())
        print("read file successful")
        self.DataSet = set_form(data)
        self.Label = set_form(self.Label)
    def readParagraph(self, path, add_label = False, sep ="\t"):
        if add_label:
            self.Label = list()
        assert self.__validPath(path)
        file = open(Util().GetDirectory() + "/DATA/" + path, "r")
        self.DataSet = list()
        lines = file.readlines()
        for line in lines:
            if len(line.strip()) > 1:
                if add_label:
                    tempData = line.strip().split(sep)
                    self.Label.append(int(tempData.pop()))
                self.DataSet.append(tempData[0])
    def separateDataSet(self, set_form, portion = 0.2, mode = "DEFAULT"):
        assert set_form in self.__SET_FORMAT
        assert self.DataSet is not None
        assert isinstance(portion, float) or isinstance(portion, float)
        assert 0.0 < portion < 1.0
        assert mode in ("DEFAULT", "SAVE", "LOAD")
        if self.Label is not None:
            trainLabel, testLabel = list(), list()
        trainData, testData = list(), list()
        Lookup_Table = Util().SplitDataSet(len(self.DataSet), portion, mode)
        for i in range(len(Lookup_Table)):
            if Lookup_Table[i] == 0:
                trainData.append(self.DataSet[i])
                if self.Label is not None:
                    trainLabel.append(self.Label[i])
            elif Lookup_Table[i] == 1:
                testData.append(self.DataSet[i])
                if self.Label is not None:
                    testLabel.append(self.Label[i])
        self.DataSet = set_form(trainData)
        if self.Label is not None:
            self.Label = set_form(trainLabel)
            return set_form(testData), set_form(testLabel)
        return testData
    def removeRedundantData(self):
        if isinstance(self.DataSet, self.LIST):
            raise TypeError("'list' object cannot do redundant")
        else:
            non_redundant = list()
            for i in range(self.DataSet.shape[1]):
                curVal = self.DataSet[1, i]
                for item in self.DataSet[:, i]:
                    if item != curVal:
                        non_redundant.append(i)
                        break
        self.DataSet = self.DataSet[:, non_redundant]
        if self.Label is not None:
            assert not isinstance(self.Label, list)
            self.Label = self.Label[non_redundant]
    def balanceDataSet(self, ratio = (0.5, 0.5)):
        assert self.DataSet is not None
        assert self.Label is not None
        assert not isinstance(self.Label, list)
        assert isinstance(ratio, tuple)
        assert len(ratio) == 2
        assert sum(ratio) == 1.0
        assert len(set(self.Label)) == 2
        sign1, sign2 = int(), int()
        firstSign = self.Label[0]
        for item in self.Label:
            if item == firstSign:
                sign1 += 1
            else: sign2 += 1
        modification = (ratio[0]*len(self.Label) - sign1, ratio[1]*len(self.Label) - sign2)
        range0 = np.nonzero(self.Label == firstSign)[0]
        range1 = np.nonzero(self.Label != firstSign)[0]
        if modification[0] > 0:
            additional0 = np.array(
                [np.random.randint(np.min(range0), np.max(range0)) for _ in range(int(modification[0]))])
            range0 = np.hstack((range0, additional0))
        elif modification[0] < 0:
            range0 = range0[np.array([np.random.randint(0, range0.size)
                                      for _ in range(int(range0.size + modification[0]))])]
        if modification[1] > 0:
            additional1 = np.array(
                [np.random.randint(np.min(range1), np.max(range1)) for _ in range(int(modification[0]))])
            range1 = np.hstack((range1, additional1))
        elif modification[1] < 0:
            range1 = range1[np.array([np.random.randint(0, range1.size)
                                      for _ in range(int(range1.size + modification[1]))])]
        totalRange = np.hstack((range0, range1))
        if isinstance(self.DataSet, list):
            r = totalRange.tolist()
            new = list()
            for item in r:
                new.append(self.DataSet[item])
            self.DataSet = new
        else: self.DataSet = self.DataSet[totalRange]
        self.Label = self.Label[totalRange]
        pass
    def convertLevelToBool(self):
        assert self.DataSet is not None
        assert self.Label is not None
        assert len(set(self.Label)) > 2
        if isinstance(self.Label, list):
            self.Label = np.array(self.Label)
        labelRange = sorted(list(set(self.Label)))
        boundary = (labelRange[0] + labelRange[-1])/2
        range0 = np.nonzero(self.Label < boundary)[0]
        range1 = np.nonzero(self.Label > boundary)[0]
        self.Label[range0] = 0
        self.Label[range1] = 1
        totalRange = np.hstack((range0, range1))
        self.Label = self.Label[totalRange]
        if isinstance(self.DataSet, list):
            r = totalRange.tolist()
            new = list()
            for item in r:
                new.append(self.DataSet[item])
            self.DataSet = new
        else: self.DataSet = self.DataSet[totalRange]
    def __curWords(self, sentence, language, Filter):
        if language is self.ENGLISH:
            words = [item.lower() for item in re.findall(language, sentence)]
        elif language is self.CHINESE:
            return [word for word in jieba.cut(" ".join(re.findall(language, sentence)))
                       if word != " "]
        words = [item for item in words if Filter(item)]
        return words
    def __generateDictionary(self, sentences):
        ret = set()
        for sentence in sentences:
            ret = ret | set(sentence)
        return list(ret)
    def __getWordExistence(self, line, dictionary):
        ret = [0] * len(dictionary)
        for word in line:
            if word in dictionary:
                ret[dictionary.index(word)] += 1
            else:
                print("The word %s does not contain in the dictionary" % (word))
        return ret
    def wordBagging(self, language, set_form, Filter = lambda x: True):
        assert set_form in self.__SET_FORMAT
        assert language in self.__LANGUAGE
        lineList = list()
        for line in self.DataSet:
            lineList.append(self.__curWords(line ,language, Filter))
        dictionary = self.__generateDictionary(lineList)
        dictMat = list()
        for line in lineList:
            dictMat.append(self.__getWordExistence(line, dictionary))
        return dictionary, set_form(dictMat)
    def head(self, value = 10):
        return self.DataSet[::value]
    def tail(self, value = 10):
        return self.DataSet[-value::]
    def pca(self, dimension):
        assert isinstance(dimension, int) and 0 < dimension <= self.DataSet.shape[1]
        meanValue = np.mean(self.DataSet, axis=0)
        data = self.DataSet - meanValue
        covariance = np.cov(data, rowvar=0)
        eigVals, eigVectors = np.linalg.eig(covariance)
        eigValsIndex = np.argsort(eigVals)[:-(dimension+1):-1]
        finalEigVectors = eigVectors[:, eigValsIndex]
        lowData = data * finalEigVectors
        reconData = (lowData * finalEigVectors.transpose()) + meanValue
        self.DataSet = reconData
        return lowData
    def graph2D(self, graphongIndexes = None, color = "#516EFF"):
        if graphongIndexes is None:
            x = np.array(self.DataSet[:, 0]).flatten()
            y = np.array(self.DataSet[:, 1]).flatten()
        elif isinstance(graphongIndexes, list) or isinstance(graphongIndexes, tuple):
            assert len(graphongIndexes) == 2
            for item in graphongIndexes:
                assert 0 <= item < self.DataSet.shape[1]
            x = np.array(self.DataSet[:, graphongIndexes[0]]).flatten()
            y = np.array(self.DataSet[:, graphongIndexes[1]]).flatten()
        if self.graph is None:
            self.__initGraph()
        graph = self.graph.add_subplot(111)
        graph.scatter(x, y, marker='o', c=color, s=20, alpha=0.2)
    def showGraph(self):
        plt.show()
