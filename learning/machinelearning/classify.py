import numpy as np
import math
import os
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import Axes3D
from learning.Helper import Util

class LogicRegression:
    def __init__(self):
        self.DataSet = None
        self.Labels = None
        self.Betas = None
        self.ErrorRate = []
    def sigmoid(self, target):
        return (1.0 / (1 + np.exp(-target)))
    def ReadSimpleFile(self, path):
        dataMat = []; labelMat = []
        fr = open(Util().GetDirectory() + "/DATA/" + path, 'r')
        for line in fr.readlines():
            lineArr = line.strip().split()
            tempLine = list()
            tempLine.append(1.0)
            length = len(lineArr) - 1
            for i in range(length):
                tempLine.append(float(lineArr[i]))
            dataMat.append(tempLine.copy())
            labelMat.append(int(lineArr[-1]))
        self.DataSet = np.array(dataMat)
        self.Labels = np.array(labelMat)
    def SeparateDataSet(self, TestSize = 0.2, Pattern = None):
        TrainData, TestData, TrainLabel, TestLabel = list(), list(), list(), list()
        if Pattern is not None:
            Lookup_Table = Pattern
        else: Lookup_Table = Util().SplitDataSet(len(self.DataSet), TestSize)
        test_index = list()
        for i in range(len(Lookup_Table)):
            if Lookup_Table[i] == 0:
                TrainData.append(list(self.DataSet[i]))
                TrainLabel.append(int(self.Labels[i]))
            elif Lookup_Table[i] == 1:
                TestData.append(list(self.DataSet[i]))
                TestLabel.append(int(self.Labels[i]))
                test_index.append(i)
        self.DataSet = np.array(TrainData)
        self.Labels = np.array(TrainLabel)
        if Pattern is None: return np.array(TestData), np.array(TestLabel)
        else: return np.array(TestData), np.array(TestLabel), test_index
    def GetDataSet(self):
        return self.DataSet, self.Labels
    def GetBetas(self):
        return self.Betas
    def ClearErrorRate(self):
        self.ErrorRate = []
    def GradientAscent(self, StepSize = 0.001, MaxCycles = 10000, if_return = False):
        DataMat = np.mat(self.DataSet)
        LabelMat = np.mat(self.Labels).transpose()
        vertical, horizontal = DataMat.shape
        Betas = np.ones((horizontal, 1))
        for i in range(MaxCycles):
            calc = self.sigmoid(DataMat * Betas)
            error = LabelMat - calc
            self.ErrorRate.append(np.abs(float(sum(error))))
            if float(np.abs(sum(error))) < 1:
                self.Betas = Betas
            Betas += StepSize * DataMat.transpose() * error
        Betas = [float(item) for item in Betas]
        if if_return: return Betas
        self.Betas = Betas
        pass
    def StochasticGradientAscent(self, StepSize = 0.01, MaxIter = 150, if_return = False):
        vertical, horizontal = self.DataSet.shape
        Betas = np.ones(horizontal)
        for iter in range(MaxIter):
            for i in range(vertical):
                Betas = 4/(1.0 + iter + i) + StepSize  #保证Betas每次循环都变
                RandIndex = int(np.random.uniform(0, vertical))
                calc = self.sigmoid(sum(self.DataSet[RandIndex] * Betas))
                error = self.Labels[RandIndex] - calc
                self.ErrorRate.append(np.abs(error))
                Betas += Betas * error * self.DataSet[RandIndex]
        if if_return: return Betas
        self.Betas = Betas
    def Predict(self, index):
        if isinstance(index, int):
            assert index >= 0 and index < len(self.DataSet)
            result = float((-self.Betas[0] - self.Betas[1] * self.DataSet[index, 1]) / self.Betas[2]) \
                     - self.DataSet[index, 2]
            if result > 0:
                return result, 1, self.Labels[index]
            else: return result, 0, self.Labels[index]
        elif len(index) == 2:
            result = float((-self.Betas[0] - self.Betas[1] * index[0]) / self.Betas[2]) - index[1]
            if result > 0:
                return result, 1
            else: return result, 0
        else:
            assert len(index) == 3
            result = float((-self.Betas[0] - self.Betas[1] * index[0]) / self.Betas[2]) - index[1]
            if result > 0:
                return result, 1, index[-1]
            else:
                return result, 0, index[-1]
    def GetLine(self):
        ret = list()
        x = np.arange(-3.0, 3.0, 0.1)
        ret.append(list(x))
        ret.append(list((-self.Betas[0]-self.Betas[1]*x)/self.Betas[2]))
        return ret
    def Graph(self, Line, name = "Logic Regression"):
        figure = plt.figure(name + " Graph & Modification Rate")
        g1 = figure.add_subplot(121)
        g1.set_title(name + " Graph")
        assert isinstance(Line, list)
        class1, class2 = [], []
        for i in range(self.Labels.size):
            if self.Labels[i] == 1: class1.append(list(self.DataSet[i]).copy())
            else: class2.append(list(self.DataSet[i]).copy())
        class1, class2 = np.array(class1), np.array(class2)
        g1.scatter(class1[:, 1], class1[:, 2], marker='o', c='#6CBAFF', s=100, alpha=0.5)
        g1.scatter(class2[:, 1], class2[:, 2], marker='o', c='#FF918B', s=100, alpha=0.5)
        g1.plot(Line[0], Line[1], color='#000000', lw=1)
        g2 = figure.add_subplot(122)
        g2.set_title("Modification Rate")
        x = [i + 1 for i in range(len(self.ErrorRate))]
        g2.plot(x, self.ErrorRate, color='#000000', lw=1)
        plt.show()
    def AdaBoost_InterFace(self, DataSet, Label, Pattern):
        if len(DataSet[0]) == 2:
            for i in range(len(DataSet)):
                DataSet[i].insert(0, 1.0)
        self.DataSet = DataSet
        self.Labels = Label
        test_data, test_label, test_index = self.SeparateDataSet(Pattern=Pattern)
        self.Betas = self.StochasticGradientAscent(if_return=True)
        error = [0] * len(test_data)
        for i in range(len(test_data)):
            result = self.Predict([test_data[i][1], test_data[i][2]])[-1]
            error[i] = int(result != test_label[i])
        return error, test_index
    def SmartTest(self, TestData, TestLabel):
        assert len(TestLabel) == len(TestData)
        PredictedLabel = list()
        for i in range(len(TestData)):
            result = self.Predict([TestData[i][1], TestData[i][2], TestLabel[i]])
            print(result)
            PredictedLabel.append(result[1])
        assert len(PredictedLabel) == len(TestLabel)
        Util().plot_ROC_curve(TestLabel, PredictedLabel)

class DecisionTree:
    def __init__(self, TreeName = "Node"):
        self.Labels = None
        self.DataSet = None
        class Node:
            def __init__(self, label, solution, index):
                self.Label = label
                self.Solution = solution
                self.Index = index
                self.Children = []
            def AddChild(self, NewLabel, NewSolution, NewIndex):
                self.Children.append(Node(NewLabel, NewSolution, NewIndex))
        self.Tree = Node(TreeName, None, -1)
    def ReadSimpleFile(self, path):
        dataMat, labelMat = list(), list()
        isinstance(path, str)
        fr = open(Util().GetDirectory() + "/DATA/" + path, "r")
        lines = fr.readlines()
        self.Labels = lines.pop(0).strip().split()
        typeid = lines.pop(0).strip().split()
        typeList = list()
        typeKind = {"int" : int, "bool" : bool, "str" : str}
        for item in typeid:
            typeList.append(typeKind[item])
        for line in lines:
            splitLine = line.strip().split(" ")
            assert len(splitLine) == len(typeList)
            tempLine = list()
            for i in range(len(splitLine)):
                if typeList[i] == bool:
                    tempLine.append(bool(int(splitLine[i])))
                else: tempLine.append(typeList[i](splitLine[i]))
            dataMat.append(tempLine.copy())
        self.DataSet = dataMat
    def SeparateDataSet(self, testSize = 0.2, pattern = None):
        assert testSize < 1.0
        if pattern is not None:
            lookeup_table = pattern
        else: lookeup_table = Util().SplitDataSet(len(self.DataSet), testSize)
        trainData, testData = list(), list()
        testIndex = list()
        for i in range(len(lookeup_table)):
            if lookeup_table[i] == 0:
                trainData.append(self.DataSet[i])
            elif lookeup_table[i] == 1:
                testData.append(self.DataSet[i])
                testIndex.append(i)
            else: raise ValueError("index out of range [0, 1]")
        self.DataSet = trainData
        if pattern is None:
            return testData
        else: return testData, testIndex
    def __GetShannonEntropy(self, DataSet):
        se = 0.0
        DataLength = len(DataSet)
        LabelCount = {}
        for line in DataSet:
            CurrentLabel = line[-1]
            LabelCount[CurrentLabel] = LabelCount.get(CurrentLabel, 0) + 1
        for key in LabelCount:
            prob = float(LabelCount[key]) / float(DataLength)
            se -= prob * math.log(prob, 2)
        return se
    def __SplitData(self, DataSet, axis, value):
        ret = []
        for line in DataSet:
            if line[axis] == value:
                NewLine = line[:axis]
                NewLine.extend(line[axis + 1:])
                ret.append(NewLine)
        return ret
    def __GetBestFeatureIndex(self, DataSet):
        FeaturesRange = len(DataSet[0]) - 1
        CurrentMaxSE, CurrentIndex = 0.0, -1
        BaseSE = self.__GetShannonEntropy(DataSet)
        for index in range(FeaturesRange):
            NewSE = 0.0
            PossibleLabels = set([line[index] for line in DataSet])
            for value in PossibleLabels:
                CurrentData = self.__SplitData(DataSet, index, value)
                prob = len(CurrentData) / float(len(DataSet))
                NewSE += prob * self.__GetShannonEntropy(CurrentData)
            InformationGain = BaseSE - NewSE
            if (InformationGain > CurrentMaxSE):
                CurrentMaxSE = InformationGain
                CurrentIndex = index
        return CurrentIndex
    def __MajoritySelection(self, DataSet):
        GrandSolution = {}
        for vote in DataSet:
            if vote not in GrandSolution.keys():
                GrandSolution[vote] = 0
            GrandSolution[vote] += 1
        GrandSolution = sorted(GrandSolution.items(), key=lambda item: item[1], reverse=True)
        return GrandSolution[0][0]
    def BuildTree(self, Tree = None, DataSet = None, DataLabels = None):
        if DataSet is None:
            DataSet = self.DataSet
        if DataLabels is None:
            DataLabels = self.Labels[:-1]
        if Tree is None:
            Tree = self.Tree
        CurrentLabels = [line[-1] for line in DataSet]  # 将所有结果遍历
        if CurrentLabels.count(CurrentLabels[0]) == len(CurrentLabels):  # 如果所有结果都一样，就分类完成
            Tree.AddChild(None, CurrentLabels[0], -1)
            return
        elif len(DataSet[0]) == 1:  # 如果没有feature可以遍历，选结果中最大的返回
            Tree.AddChild(None, self.__MajoritySelection(DataSet), -1)
            return
        else:
            BestFeature = self.__GetBestFeatureIndex(DataSet)  # 返回最好features的index
            BestLabel = DataLabels[BestFeature]  # 得到最好features那行的label
            del (DataLabels[BestFeature])  # 删除这次遍历最好的label，为下一层传参做准备
            FeatureValues = set([line[BestFeature] for line in DataSet])  # 遍历最好features的那行的所有可能
            for value in FeatureValues:  # 对可能进行遍历
                NewLabels = DataLabels[:]  # 复制上上上行的label，作为下次递归的参数
                Tree.AddChild(BestLabel, value, self.Labels.index(BestLabel))  # 在本树里开辟一个新的孩子，存入目前最好的label以及当前的可能
                self.BuildTree(Tree.Children[-1], self.__SplitData(DataSet, BestFeature, value), NewLabels)  # 刚刚创建的孩子作为下一个遍历的对象
        self.Tree = Tree
    def VisualizeTree(self, tree = None, tabcount = 0):
        if tree is None:
            tree = self.Tree
        for i in range(tabcount): print('\t', end='')
        if tree.Solution == None:
            print("tree name: %s" % tree.Label)
        elif tree.Label == None:
            print("result: %s" % tree.Solution)
        else:
            print("if '%s' = " % (tree.Label), tree.Solution, " -> " + str(tree.Index), sep='')
        if len(tree.Children) == 0:
            return
        tabcount += 1
        for item in tree.Children:
            self.VisualizeTree(item, tabcount)
        return
    def Predict(self, datas, tree = None):
        if tree is None:
            tree = self.Tree
        assert isinstance(datas, list)
        if tree.Label is None:
            return tree.Solution
        for options in tree.Children:
            if datas[options.Index] == options.Solution or options.Index == -1:
                return self.Predict(datas, options)
        return None
    def SmartTest(self, testSet):
        predicts = list()
        for line in testSet:
            assert len(line) == len(self.Labels)
            predict = self.Predict(line)
            if predict is None:
                predicts.append(0)
            else:
                predicts.append(int(line[-1] == predict))
        error = float(len(predicts) - sum(predicts)) / len(predicts)
        return error, predicts

class SupportVectorMachine:
    def __init__(self):
        self.DataSet = None
        self.Labels = None
        self.Betas = None
        self.Constant = None
        self.Weights = None
        self.ValidElements = None
        self.ErrorsStorage = None  #will allocate memory for storing the data when enable capacity
        self.KernelValues = None   #will allocate memory when using kernel (solving non-linear machine)
    def GetDataSet(self):
        return self.DataSet
    def GetLabels(self):
        return self.Labels
    def GetBetas(self):
        return self.Betas
    def ClearKernel(self):
        self.KernelValues = None
    def GetLine(self, if_return = False):
        vertical, horizontal = self.DataSet.shape
        weights = np.zeros((horizontal, 1))
        for i in range(vertical):
            weights += np.multiply(self.Betas[i] * self.Labels[i], self.DataSet[i].T)
        self.Weights =  weights
        if if_return == True:
            return self.Weights
    def SeparateDataSet(self, TestSize = 0.2, mode = "DEFAULT"):
        assert mode in ("LOAD", "SAVE", "DEFAULT")
        TrainData, TrainLabel, TestData, TestLabel = list(), list(), list(), list()
        Lookup_Table = Util().SplitDataSet(len(self.DataSet), TestSize, mode)
        for i in range(len(Lookup_Table)):
            if Lookup_Table[i] == 0:
                TrainData.append(self.DataSet[i])
                TrainLabel.append(self.Labels[i])
            elif Lookup_Table[i] == 1:
                TestData.append(self.DataSet[i].tolist()[0])
                TestLabel.append(int(self.Labels[i]))
        self.DataSet = np.mat(np.array(TrainData))
        self.Labels = np.mat(np.array(TrainLabel)).transpose()
        return TestData, TestLabel
    def Predict(self, value, given_label = None):
        if isinstance(value, int):
            assert value < len(self.DataSet) and value >= 0
            if self.KernelValues is None:
                ret = self.DataSet[value]*np.mat(self.Weights) + self.Constant
            else:
                k = self.CalculateKernel(self.DataSet[value], "non-linear", 1.3, self.DataSet[self.ValidElements])
                ret = k.T * np.multiply(self.Labels[self.ValidElements], self.Betas[self.ValidElements]) + self.Constant
            if ret < 0:
                return float(ret), -1 , int(self.Labels[value])
            else: return float(ret), 1 , int(self.Labels[value])
        elif value.size == self.DataSet.shape[1]:
            if self.KernelValues is None:
                ret = np.mat(value)*np.mat(self.Weights) + self.Constant
            else:
                k = self.CalculateKernel(np.mat(value), "non-linear", 1.3, self.DataSet[self.ValidElements])
                ret = k.T * np.multiply(self.Labels[self.ValidElements], self.Betas[self.ValidElements]) + self.Constant
            if isinstance(given_label, int):
                if ret < 0:
                    return float(ret), -1, given_label
                else:
                    return float(ret), 1, given_label
            if ret < 0:
                return float(ret), -1
            else: return float(ret), 1
    def ReadSimpleFile(self, path):
        assert isinstance(path, str)
        if path[-4::] != '.txt':
            print('Read file only support txt format')
            return None
        if not os.path.exists(Util().GetDirectory() + "/DATA/" + path):
            print('File does not exist: %s' % (path))
            return None
        file = open(Util().GetDirectory() + "/DATA/" + path, 'r')
        try:
            lines = file.readlines()
            RawData, DataSet, Labels = None, [], []
            for line in lines:
                RawData = line.strip().split('\t')
                Labels.append(int(float(RawData.pop())))
                DataSet.append([float(item) for item in RawData.copy()])
        except IndexError and ValueError and KeyError:
            print('invalid file arrangement'); return None
        except:
            print('unknown error'); return None
        else:
            print('Read file successful')
        self.DataSet = np.mat(DataSet)
        self.Labels = np.mat(Labels).transpose()
    def Platt_SMO(self, constrain, tolerance, max_iter, capacity = "ENABLED"):
        def CalculateError(j, update = False):
            if self.KernelValues is not None:
                calc_j = float(np.multiply(betas, LabelMat).T * self.KernelValues[:, j]) + constant
            else: calc_j = float(np.multiply(betas, LabelMat).T * (DataMat * DataMat[j].T)) + constant
            if update:
                self.ErrorsStorage[j] = [1, calc_j - float(LabelMat[j])]
                return
            return calc_j - float(LabelMat[j])
        def SelectBestPair(i, error_i):
            if capacity != "ENABLED":
                ret = Util().SelectRandomItem(i, vertical)
                return ret, CalculateError(ret)
            else:
                assert self.ErrorsStorage.shape
                best, modification_best, error_best = -1, 0, 0
                self.ErrorsStorage[i] = [1, error_i]
                ValidList = np.nonzero(self.ErrorsStorage[:, 0].A)[0]
                if len(ValidList) > 1:
                    for item in ValidList:
                        if item == i:
                            continue
                        error_item = CalculateError(item)
                        modification_item = np.abs(error_i - error_item)
                        if modification_item > modification_best:
                            best = item
                            modification_best = modification_item
                            error_best = error_item
                    return best, error_best
                else:
                    ret = Util().SelectRandomItem(i, vertical)
                    return ret, CalculateError(ret)
        def Train(i):
            nonlocal constant
            i_error = CalculateError(i)
            if (LabelMat[i] * i_error < -tolerance and betas[i] < constrain) or \
                    (LabelMat[i] * i_error > tolerance and betas[i] > 0):  #计算偏移初始条件
                j, j_error = SelectBestPair(i, i_error)
                i_old = betas[i].copy(); j_old = betas[j].copy()
                if LabelMat[j] != LabelMat[i]:
                    min_difference = max(0, betas[j] - betas[i])
                    max_difference = min(constrain, constrain + betas[j] - betas[i])
                else:
                    min_difference = max(0, betas[i] +  betas[j] - constrain)
                    max_difference = min(constrain, betas[i] + betas[j])
                if max_difference == min_difference:            #第一个运算结束条件判断
                    print("BC1: equal difference, i: %d, j: %d"%(i, j))
                    return 0
                if self.KernelValues is not None:    #如果是在高次svm里调用执行这个
                    eta = 2.0 * self.KernelValues[i, j] - self.KernelValues[i, i] - self.KernelValues[j, j]
                else: eta = 2.0 * (DataMat[i] * DataMat[j].T) - (DataMat[i] * DataMat[i].T) - (DataMat[j] * DataMat[j].T)
                if eta >= 0:                                    #第二个运算结束条件判断
                    print("BC2: eta >= 0, i: %d, j: %d"%(i, j))
                    return 0
                betas[j] -= LabelMat[j] * (i_error - j_error)/eta
                betas[j] = Util().ClipStepSize(max_difference, betas[j], min_difference)
                if capacity == "ENABLED":
                    CalculateError(j, True)
                if np.abs(betas[j] - j_old) < pow(10, -5):      #第三个运算结束条件判断
                    print("BC3: j_offset is less than 10^-5, i: %d, j: %d"%(i, j))
                    return 0
                betas[i] += LabelMat[i] * LabelMat[j] * (j_old - betas[j])  #反方向移动
                if capacity == "ENABLED":
                    CalculateError(i, True)
                if self.KernelValues is not None:    #如果是在高次svm里调用执行这个
                    i_constant = constant - i_error - LabelMat[i] * (betas[i] - i_old) * self.KernelValues[i, i] - \
                                 LabelMat[j] * (betas[j] - j_old) * self.KernelValues[i, j]
                    j_constant = constant - i_error - LabelMat[i] * (betas[i] - i_old) * self.KernelValues[i, j] - \
                                 LabelMat[j] * (betas[j] - j_old) * self.KernelValues[j, j]
                else:
                    i_constant = constant - i_error - LabelMat[i] * (betas[i] - i_old) * (DataMat[i] * DataMat[i].T) - \
                                 LabelMat[j] * (betas[j] - j_old) * (DataMat[i] * DataMat[j].T)
                    j_constant = constant - j_error - LabelMat[i] * (betas[i] - i_old) * (DataMat[i] * DataMat[j].T) - \
                                 LabelMat[j] * (betas[j] - j_old) * (DataMat[j] * DataMat[j].T)
                if 0 < betas[i] and constrain > betas[i]: constant = i_constant
                elif 0 < betas[j] and constrain > betas[j]: constant = j_constant
                else: constant = (i_constant - j_constant)/2 + j_constant
                return 1
            else: return 0
        DataMat = self.DataSet
        LabelMat = self.Labels
        constant = int()
        vertical, horizontal = DataMat.shape
        if capacity == "ENABLED":
            self.ErrorsStorage = np.mat(np.zeros((vertical, 2)))
        betas = np.mat(np.ones((vertical, 1)))
        iter = 0
        TraverseAll = True
        BetasPairsChanged = 0
        if capacity != "ENABLED":
            while iter < max_iter:
                BetasPairsChanged = 0
                for i in range(vertical):
                    BetasPairsChanged += Train(i)
                    print("iter: %d\ti: %d\tpairs_changed: %d" % (iter, i, BetasPairsChanged))
                if BetasPairsChanged == 0:
                    iter += 1
                else: iter = 0
            print("iteration number: %d" % iter)
        else:
            while iter < max_iter and (BetasPairsChanged > 0 or TraverseAll == True):
                BetasPairsChanged = int()
                if TraverseAll == True:
                    for i in range(vertical):
                        BetasPairsChanged += Train(i)
                        print("full_set, iter: %d\ti: %d\tpairs_changed: %d"%(iter, i, BetasPairsChanged))
                    iter += 1
                else:
                    ValidElements =np.nonzero((betas.A > 0) * (betas.A < constrain))[0]
                    for i in ValidElements:
                        BetasPairsChanged += Train(i)
                        print("non_bound, iter: %d\ti: %d\tpairs_changed: %d" % (iter, i, BetasPairsChanged))
                    iter += 1
                if TraverseAll == True:
                    TraverseAll = False
                elif BetasPairsChanged == 0:
                    TraverseAll = True
                print("iteration number: %d" % iter)
        self.Betas = betas
        self.Constant = constant
        self.ValidElements = np.nonzero(np.array(self.Betas) > 0)[0]
    def CalculateKernel(self, current, dimension, sig, given_data=None):
        assert dimension in ["linear", "non-linear"]
        if given_data is None:
            kernel = np.mat(np.zeros((self.DataSet.shape[0], 1)))
            if dimension == "linear":
                    kernel = self.DataSet * current.transpose()
            elif dimension == "non-linear":
                for line in range(self.DataSet.shape[0]):
                    offset = self.DataSet[line] - current
                    kernel[line] = offset * offset.T
                kernel = np.exp(kernel / -1 * sig ** 2)
                pass
        else:
            kernel = np.mat(np.zeros((given_data.shape[0], 1)))
            if dimension == "linear":
                kernel = given_data * current.transpose()
            elif dimension == "non-linear":
                for line in range(given_data.shape[0]):
                    offset = given_data[line] - current
                    kernel[line] = offset * offset.T
                kernel = np.exp(kernel / -1 * sig ** 2)
        return kernel
    def RadicalBias_Gaussian(self, constrain, tolerance, max_iter, capacity = "ENABLED", dimension = "non-linear", sigma = 1.3):
        vertical = self.DataSet.shape[0]
        if self.KernelValues == None:
            self.KernelValues = np.mat(np.zeros((vertical, vertical)))
        for i in range(vertical):
            self.KernelValues[:, i] = self.CalculateKernel(self.DataSet[i], dimension, sigma)
        self.Platt_SMO(constrain, tolerance, max_iter, capacity)
    def GraphPoints(self):
        group_1, group_2, group_3 = [[], []], [[], []], [[], []]
        for i in range(len(self.Labels)):
            if i in self.ValidElements:
                group_3[0].append(float(self.DataSet[i, 0]))
                group_3[1].append(float(self.DataSet[i, 1]))
            elif self.Labels[i] == 1:
                group_1[0].append(float(self.DataSet[i, 0]))
                group_1[1].append(float(self.DataSet[i, 1]))
            else:
                group_2[0].append(float(self.DataSet[i, 0]))
                group_2[1].append(float(self.DataSet[i, 1]))
        if False and len(self.Weights) != 0:     #开发失败 fix 这里false的原因是不让程序触发
            xMin, xMax = float(min(self.DataSet[:, 0])), float(max(self.DataSet[:, 0]))
            yMin, yMax = float(min(self.DataSet[:, 1])), float(max(self.DataSet[:, 1]))
            assert xMax > xMin
            x, y = list(), list()
            step_size = (xMax - xMin)/5
            while xMax > xMin:
                y_val = xMin*self.Weights[0] + self.Weights[1]
                if y_val > yMin and y_val < yMax:
                    x.append(xMin)
                    y.append(y_val)
                xMin += step_size
            if len(x) < 2:
                print("invalid line")
                return
            x = np.array(x)
            y = np.array(y)
            plt.plot(x, y, color='#000000', lw=2)
        if len(group_3[0]) != 0:
            plt.scatter(group_3[0], group_3[1], marker='o', c='#000000', s=100, alpha=0.5)
        plt.scatter(group_1[0], group_1[1], marker='o', c='#6CBAFF', s=100, alpha=0.5)
        plt.scatter(group_2[0], group_2[1], marker='o', c='#FF918B', s=100, alpha=0.5)
        plt.show()

class NaiveBayes:
    def __init__(self):
        self.DataSet = None
        self.Labels = None
        self.Dictionary = None
        self.WordMat = None
        self.DecisionBoundary = None
        self.ResultLabels = None
        self.UsingEnglish = True
    def ReadSimpleFile(self, path):
        assert isinstance(path, str)
        if path[-4::] != '.txt':
            print('Read file only support txt format'); return None
        if not os.path.exists(Util().GetDirectory() + "/DATA/" + path):
            print('File does not exist: %s'%(path)); return None
        file = open(Util().GetDirectory() + "/DATA/" + path, 'r')
        try:
            Contents, Labels = list(), list()
            lines = file.readlines()
            self.ResultLabels = lines.pop(0).replace("\n", "").split(" ")
            for line in lines:
                line = line.replace("\n", "").split("\t")
                if float(line[-1]) > 3.0:
                    Labels.append(1)
                    Contents.append(line[0])
                elif float(line[-1]) < 3.0:
                    Labels.append(0)
                    Contents.append(line[0])
        except IndexError and ValueError and KeyError: print('invalid file arrangement'); return None
        except: print('unknown error'); return None
        else: print('Read file successful')
        self.DataSet = Contents
        self.Labels = Labels
    def SeparateDataSet(self, TestSize = 0.2, mode = "DEFAULT", if_return = False):
        assert mode in ("LOAD", "SAVE", "DEFAULT")
        TrainData, TrainLabel, TestData, TestLabel = list(), list(), list(), list()
        Lookup_Table = Util().SplitDataSet(len(self.DataSet), TestSize, mode)
        for i in range(len(Lookup_Table)):
            if Lookup_Table[i] == 0:
                TrainData.append(self.DataSet[i])
                TrainLabel.append(self.Labels[i])
            elif Lookup_Table[i] == 1:
                TestData.append(self.DataSet[i])
                TestLabel.append(int(self.Labels[i]))
        self.DataSet = np.array(TrainData)
        self.Labels = np.array(TrainLabel)
        if if_return:
            return TestData, TestLabel, np.nonzero(np.array(Lookup_Table) == 1)[0]
        return TestData, TestLabel
    def CutWords(self, Sentences = None, isEnglish = True):
        if isEnglish:
            if Sentences is None:
                Sentences = self.DataSet
            mode = re.compile(r'[a-z|A-Z]+')
            ret = list()
            if isinstance(Sentences, str):
                words = Sentences.split(" ")
                for word in words:
                    if word.isalpha() and len(word) > 2:
                        ret.append(word.lower())
            else:
                for line in Sentences:
                    current = list()
                    words = [item.lower() for item in re.findall(mode, line)]
                    for word in words:
                        if len(word) > 2:
                            current.append(word)
                    ret.append(current.copy())
        else:
            if Sentences is None:
                Sentences = self.DataSet
            import jieba
            mode = re.compile(r'([\u4E00-\u9FA5]+|\w+)')
            if isinstance(Sentences, str):
                ret = [word for word in jieba.cut(" ".join(re.findall(mode, Sentences)))
                       if word != " "]
            else:
                ret = []
                assert isinstance(Sentences, list)
                for line in Sentences:
                    ret.append([word for word in jieba.cut(" ".join(re.findall(mode, line)))
                                if word != " "])
        if Sentences is None:
            self.DataSet = ret
            return
        return ret
    def CreateDictionary(self):
        DataSet = self.DataSet
        ret = set([])
        for line in DataSet:
            ret = ret | set(self.CutWords(line, self.UsingEnglish))
        self.Dictionary = list(ret)
    def __GetWordExistence(self, DataSet):
        if isinstance(DataSet, str):
            DataSet = self.CutWords(DataSet, self.UsingEnglish)
        ret = [0] * len(self.Dictionary)
        for line in DataSet:
            if line in self.Dictionary:
                ret[self.Dictionary.index(line)] += 1
            else: pass#print("The word %s does not contain in the dictionary" % (line))
        return ret
    def BuildModel(self):
        # self.WordMat = list()
        # for line in self.DataSet:
        #     self.WordMat.append(self.__GetWordExistence(line))
        # self.WordMat = np.mat(self.WordMat)
        vertical, horizontal = self.WordMat.shape
        pFirstClass = sum(self.Labels) / float(vertical)
        firstClassExistence = np.mat(np.ones(horizontal))
        secondClassExistence = np.mat(np.ones(horizontal))
        firstWordCount, secondWordCount = 2.0, 2.0
        for i in range(vertical):
            if self.Labels[i] == 1:
                firstClassExistence += self.WordMat[i]
                firstWordCount += sum(self.WordMat[i])
            else:
                secondClassExistence += self.WordMat[i]
                secondWordCount += sum(self.WordMat[i])
        a = 10
        self.DecisionBoundary = {
            "FirstClassBoundary" : np.log(firstClassExistence/firstWordCount),
            "SecondClassBoundary" : np.log(secondClassExistence/secondWordCount),
            "FirstClassProb" : pFirstClass
        }
    def Predict(self, inputs):
        p1 = float(inputs * self.DecisionBoundary['FirstClassBoundary'].transpose()) + np.log(self.DecisionBoundary['FirstClassProb'])
        p2 = float(inputs * self.DecisionBoundary['SecondClassBoundary'].transpose()) + np.log(1.0 - self.DecisionBoundary['FirstClassProb'])
        if p1 > p2:
            return self.ResultLabels[1], p1
        else: return self.ResultLabels[0], p2
    def SmartTest(self, testSet, words, testLabel = None):
        error = int()
        predicts = list()
        for i in range(len(testSet)):
            result, p = self.Predict(testSet[i])
            predicts.append(result)
            if predicts[i] != self.ResultLabels[testLabel[i]]:
                error += 1
            print(words[i])
            print("predict: %s, actual: %s\tparam: %.6f\n\n"%(predicts[i], self.ResultLabels[testLabel[i]], p))
        return error / float(len(testSet)), predicts

class knn:
    def __init__(self):
        self.DataSet = None
        self.Label = None
        self.LabelName = list()
        self.DataRange = [-np.inf, np.inf]
        self.USING_SAMPLE_NUM = -1
        self.ColorCode = [
            "#5C9EFF", "#FF5CE7", "#FFB15C", "#AEFF5C",
            "#695CFF", "#FF825C", "#5CFFE6", "#FFD65C",
            "#5CFF70", "#FFF85C", "#FF5C74", "#AD5CFF"
        ]
    def ReadSimpleFile(self, path):
        assert isinstance(path, str)
        if path[-4::] != '.txt':
            print('Read file only support txt format')
            return None
        if not os.path.exists(Util().GetDirectory() + "/DATA/" + path):
            print('File does not exist: %s' % (path))
            return None
        file = open(Util().GetDirectory() + "/DATA/" + path, 'r')
        try:
            lines = file.readlines()
            RawData, DataSet, Labels = None, list(), list()
            for line in lines:
                RawData = line.strip().split('\t')
                newLabel = RawData.pop()
                if not newLabel in self.LabelName:
                    self.LabelName.append(newLabel)
                Labels.append(newLabel)
                DataSet.append([float(item) for item in RawData.copy()])
        except IndexError and ValueError and KeyError:
            print('invalid file arrangement'); return None
        except:
            print('unknown error'); return None
        else:
            print('Read file successful')
        self.DataSet = np.array(DataSet)
        self.Labels = np.array(Labels, dtype=np.str).transpose()
        pass
    def SeparateDataSet(self, TestSize = 0.2, mode = "DEFAULT"):
        assert mode in ("LOAD", "SAVE", "DEFAULT")
        TrainData, TrainLabel, TestData, TestLabel = list(), list(), list(), list()
        Lookup_Table = Util().SplitDataSet(len(self.DataSet), TestSize, mode)
        for i in range(len(Lookup_Table)):
            if Lookup_Table[i] == 0:
                TrainData.append(self.DataSet[i])
                TrainLabel.append(self.Labels[i])
            elif Lookup_Table[i] == 1:
                TestData.append(self.DataSet[i])
                TestLabel.append(self.Labels[i])
        self.DataSet = np.array(TrainData)
        self.USING_SAMPLE_NUM = self.DataSet.shape[0]//2
        self.Labels = np.array(TrainLabel, dtype=np.str).transpose()
        return np.array(TestData), np.array(TestLabel, dtype=np.str)
    def Normolization(self):
        self.DataRange[0], self.DataRange[1] = self.DataSet.min(0), self.DataSet.max(0)
        ret = np.zeros((self.DataSet.shape))
        vertical = self.DataSet.shape[0]
        ret = self.DataSet - np.tile(self.DataRange[0], (vertical, 1))
        ret /= np.tile((self.DataRange[1] - self.DataRange[0]), (vertical, 1))
        self.DataSet = ret
    def __Classify(self, target):
        vertical = self.DataSet.shape[0]
        targetMat = np.tile(target, (vertical, 1))- self.DataSet
        distanceSum = (targetMat ** 2).sum(axis=1)
        distanceSum = distanceSum ** 0.5
        distanceIndex = distanceSum.argsort()
        labelCount = dict()
        for i in range(self.USING_SAMPLE_NUM):
            tempLabel = self.Labels[distanceIndex[i]]
            labelCount[tempLabel] = labelCount.get(tempLabel, 0) + 1
        labelCount = sorted(labelCount.items(), key= lambda item: item[1], reverse=True)
        return labelCount[0][0]
    def Predict(self, target):
        return self.__Classify(np.array(target))
    def SmartTest(self, testData, testLabel, toggle_print=False):
        vertical = testData.shape[0]
        error = 0
        for i in range(vertical):
            result = self.Predict(testData[i])
            if toggle_print:
                print("predict: %s, real: %s"%(result, testLabel[i]))
            if result != testLabel[i]:
                error += 1
        return (vertical-error)/float(vertical)
    def graph(self, graphLabels):
        figure = plt.figure()
        if isinstance(graphLabels, int):
            assert 0 <= graphLabels < self.DataSet.shape[1] - 1
            curGraph = figure.add_subplot(111)
        else:
            assert len(graphLabels) == len(self.LabelName) - 1
            curGraph = figure.add_subplot(111, projection='3d')
        for i in range(len(self.LabelName)):
            currentIndex = np.nonzero(self.Labels == self.LabelName[i])[0]
            if isinstance(graphLabels, int):
                curGraph.scatter(self.DataSet[currentIndex, graphLabels], self.DataSet[currentIndex, -1],
                           c=self.ColorCode[i], marker='o', s=100, alpha=0.2)
            else:
                curGraph.scatter(self.DataSet[currentIndex, graphLabels[0]],
                                 self.DataSet[currentIndex, graphLabels[1]],
                                 self.DataSet[currentIndex, -1],
                                 c=self.ColorCode[i], marker='o', s=100, alpha=0.2)
        plt.show()