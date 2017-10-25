import numpy as np
from Helper import Util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Node:
    """
    Node is the data structure of tree_based regression
    """
    def __init__(self, index, value, left, right):
        self.Index = index          #int, store the best split index
        self.Value = value          #np.mat, store Betas
        self.leftNode = left        #<class>Node, store left child
        self.rightNode = right      #<class>Node, store right child

class Regression:
    def __init__(self):
        self.DataSet = None             #np.mat, Data matrix
        self.Labels = None              #np.mat, correspond value of each data set
        self.Title = None               #list, the title(name) of each vertical data
        self.independentVar = None      #list, column that user uses for regression
        self.Betas = None               #list, beta values after regression
        self.Alpha = 0.0               #float, alpha constant for linear regression (bias)
        self.r_square = None            #float, r square for test data
        self.Tree = None                #class<Node>, tree for tree based regression
        self.ITER_NUM = 30              #iter number for shrinking
        self.MIN_NUM_DATA = 5           #min number of data set for each decision leaf
        self.MIN_DELTA_ERR = 2          #min error modification for decision leaf
        self.RIDGE_REGRESSION = 1       #regression method
        self.STAGEWISE_REGRESSION = 2   #regression method
        self.TREE_BASED = 3             #regression method
        self.REG_METHOD = (self.RIDGE_REGRESSION, self.STAGEWISE_REGRESSION)
    def ReadSimpleFile(self, path):
        """
        the function reads data from a .txt file. The file should follow the format:
            1) data sepa
        :param path: str, the FILE NAME of the target file.
        :return: None
        """
        assert isinstance(path, str)
        dataMat, labelMat = list(), list()
        fr = open(Util().GetDirectory() + "/DATA/" + path, 'r')
        lines = fr.readlines()
        self.Title = lines.pop(0).strip().split(" ")
        for line in lines:
            splitLine = line.strip().split("\t")
            length = len(splitLine) - 1
            if length != len(self.Title):
                continue
            tempLine = list()
            for i in range(length):
                tempLine.append(float(splitLine[i]))
            dataMat.append(tempLine.copy())
            labelMat.append(float(splitLine[-1]))
        self.DataSet = np.array(dataMat)
        self.Labels = np.array(labelMat)
    def GetFunction(self):
        """
        Get alpha value and beta values
        :return: alpha value (float), beta values (list)
        """
        assert len(self.independentVar) == len(self.Betas)
        return self.Alpha, [self.independentVar, self.Betas]
    def SeparateDataSet(self, testSize = 0.2, pattern = None):
        """
        The function separates data set to train data and test data. train data stores in the object.
        :param testSize: float in (0.0, 10.0), the portion of test data. default value is 0.2.
        :param pattern: the pattern for splitting, usually using list from MachineLearningHelper.Util.SplitDataSet
                        None will randomly split.
        :return: testData (ndarray), testLabel (nd.array)
        """
        assert testSize < 1.0 and testSize > 0.0
        if pattern is None:
            lookeup_table = Util().SplitDataSet(self.DataSet.shape[0], testSize)  #get flag of splitting index
        else: lookeup_table = pattern
        trainData, trainLabel, testData, testLabel = list(), list(), list(), list()
        testIndex = list()
        for i in range(len(lookeup_table)):
            if lookeup_table[i] == 0:
                trainData.append(self.DataSet[i])
                trainLabel.append(self.Labels[i])
            elif lookeup_table[i] == 1:
                testData.append(self.DataSet[i])
                testLabel.append(self.Labels[i])
                testIndex.append(i)
            else: raise ValueError("index out of range [0, 1]")
        self.DataSet = np.array(trainData)
        self.Labels = np.array(trainLabel)
        if pattern is None:
            return np.array(testData), np.array(testLabel)
        else: return np.array(testData), np.array(testLabel), testIndex
    def LinearRegression(self, features = None, mode = "DEFAULT", test_data = None, test_label = None):
        assert (mode in ("DEFAULT", "LSE") or mode in self.REG_METHOD or mode == self.TREE_BASED)
                                                                #LSE: Least Square Method
        if mode == "LSE":
            assert isinstance(features, int)
            assert features >= 0 and features < len(self.Title)
            return self.__LeastSquareMethod(features)
        contain = list()
        if features is not None:
            assert isinstance(features, list) or isinstance(features, tuple)
            if isinstance(features[0], bool):
                assert len(features) == len(self.Title)
                for i in range(len(features)):
                    if features[i]: contain.append(i)
            elif isinstance(features[0], int):
                features = set(features)
                for item in features:
                    assert item <= len(self.Title)
                    contain.append(item)
            elif isinstance(features[0], str):
                features = set(features)
                for item in features:
                    contain.append(self.Title.index(item))
            else: raise TypeError("invalid type for features")
        else:
            contain = [item for item in range(len(self.Title))]
        self.independentVar = sorted(contain)
        xData, xMean = list(), list()
        xData = np.mat(self.DataSet[:, self.independentVar])
        for i in range(xData.shape[1]):
            xMean.append(np.mean(xData[:, i]))
        if mode in self.REG_METHOD:
            return self.__Shrinking(xData, xMean, test_data, test_label, mode)
        if mode == self.TREE_BASED:
            return self.__TreeBasedRegression(xData)
        betas = np.dot(np.linalg.inv(np.dot(np.transpose(xData), xData)),
                            np.dot(np.transpose(xData), np.mat(self.Labels).T))
        alpha = np.mean(np.mat(self.Labels))
        for i in range(len(xMean)):
            alpha -= xMean[i] * betas[i]
        self.Betas = [float(item) for item in betas]
        self.Alpha = float(alpha)
    def __Shrinking(self, xData, xMean, testD, testL, mode):
        def GetRidgeWeight(param = 0.2):
            xVector = xDifference.transpose() * xDifference
            xVector += np.eye(xDifference.shape[1]) * param
            if np.linalg.det(xVector) == 0.0:
                print("singular matrix, cannot do inverse"); return
            weight = xVector.I * (xDifference.transpose() * yDifference.transpose())
            return weight
        def GetStageWise(stepSize = 0.01, iterTime = 100):
            pass  #fix in future
        yMean = np.mat(np.repeat(np.mean(self.Labels), len(self.Labels)))
        yDifference = np.mat(self.Labels) - yMean
        xDifference = (xData - np.mat([xMean for i in range(xData.shape[0])])) / np.var(xData, 0)  #generator fix later
        assert isinstance(self.ITER_NUM, int)
        weights = np.mat(np.zeros((self.ITER_NUM, xData.shape[1])))
        for i in range(self.ITER_NUM):
            if mode == self.RIDGE_REGRESSION:
                weights[i, :] = GetRidgeWeight(np.exp(i-10)).transpose()
            elif mode == self.STAGEWISE_REGRESSION:
                weights[i, :] = GetStageWise().transpose()
        #starts cross validation here
        bestWeight = weights[0]
        self.r_square = 0.0
        for weight in weights:
            self.Betas = weight
            newR, predicted = self.SmartTest(testD, testL)
            if newR < 0:
                if newR < self.r_square:
                    self.r_square = newR
                    bestWeight = weight
            elif newR > self.r_square:
                self.r_square = newR
                bestWeight = weight
        self.Betas = bestWeight
    def __LeastSquareMethod(self, feature):
        xData = np.mat(self.DataSet[:, feature])
        xMean = np.mat(np.repeat(np.mean(xData), xData.shape[1]))
        yMean = np.mat(np.repeat(np.mean(self.Labels), len(self.Labels)))
        xDifference = xData - xMean
        yDifference = np.mat(self.Labels) - yMean
        beta = (xDifference * yDifference.transpose()) / (xDifference * xDifference.transpose())
        alpha = float(yMean[0, [0]]) - float(beta) * float(xMean[0, 0])
        self.Betas = [float(beta), ]
        self.Alpha = float(alpha)
    def __TreeBasedRegression(self, data):
        def __splitbyValue(data, index, value):
            left = data[np.nonzero(data[:, index] > value)[0]]
            right = data[np.nonzero(data[:, index] <= value)[0]]
            return left, right
        def __chooseBestSplit(data):
            if len(set(data[:, -1].transpose().tolist()[0])) == 1:
                return None, __leafModel(data)
            vertical, horizontal = data.shape
            curErr = __treeModel(data)
            smallestErr, bestIndex, bestValue = np.inf, -1, -1
            for choose in range(horizontal - 1):
                for value in set(data[:, choose].T.tolist()[0]):
                    leftChild, rightChild = __splitbyValue(data, choose, value)
                    if leftChild.shape[0] <= self.MIN_NUM_DATA or rightChild.shape[0] <= self.MIN_NUM_DATA:
                        continue
                    newError = __treeModel(leftChild) + __treeModel(rightChild)
                    if newError < smallestErr:
                        bestIndex = choose
                        bestValue = value
                        smallestErr = newError
            if curErr - smallestErr < self.MIN_DELTA_ERR:
                return None, __leafModel(data)
            left, right = __splitbyValue(data, bestIndex, bestValue)
            if left.shape[0] < self.MIN_NUM_DATA or right.shape[0] < self.MIN_NUM_DATA:
                return None, __leafModel(data)
            return bestIndex, bestValue
        def __treeSpanning(data):
            index, value = __chooseBestSplit(data)
            tree = Node(index, value, None, None)
            if tree.Index is None:
                return tree
            leftData, rightData = __splitbyValue(data, tree.Index, tree.Value)
            tree.leftNode = __treeSpanning(leftData)
            tree.rightNode = __treeSpanning(rightData)
            return tree
        def __LinearSolve(data):
            vertical, horizontal = data.shape
            xData = np.mat(np.ones((vertical, horizontal)))
            yData = np.mat(np.ones((vertical, 1)))
            xData[:, 1:horizontal] = data[:, 0:horizontal-1]
            yData = data[:, -1]
            xVar = xData.transpose() * xData
            betas = xVar.I * (xData.transpose() * yData)
            return betas, xData, yData
        def __leafModel(data):
            betas, x, y = __LinearSolve(data)
            return betas
        def __treeModel(data):
            betas, x, y = __LinearSolve(data)
            predict = x * betas
            return sum(np.power(y - predict, 2))
        if self.Labels.ndim == 1:
            label = np.array(np.mat(self.Labels))
        else: label = self.Labels
        data = np.mat(np.concatenate((data, label.transpose()), axis=1))
        self.Tree = __treeSpanning(data)
        return
    def Predict(self, inputs, add_weight = None):
        if self.Tree is not None:
            return self.__treeBasedPredict(inputs)
        if not isinstance(inputs, list) or isinstance(inputs, tuple):
            try: inputs = inputs.tolist()
            except: raise TypeError("invalid type")
        assert len(inputs) == len(self.Betas)
        predict = self.Alpha
        if add_weight is not None:
            assert isinstance(add_weight, float)
            vertical = self.DataSet.shape[0]
            weights = np.mat(np.eye((vertical)))
            xData = np.mat(self.DataSet[:, self.independentVar])
            for i in range(vertical):
                difference  = inputs - xData[i]
                weights[i, i] = np.exp(difference * difference.T / (-2.0*add_weight**2))
            xVector = xData.T * (weights * xData)
            if np.linalg.det(xVector) == 0.0:
                print("xVector is singular matrix, cannot do inverse"); return
            betas = xVector.I * (xData.T * (weights * np.mat(self.Labels).T))
            for i in range(len(inputs)):
                predict += betas[i] * inputs[i]
            predict = float(predict)
        else:
            for i in range(len(inputs)):
                predict += float(self.Betas[i]) * inputs[i]
        return predict
    def __treeBasedPredict(self, inputs):
        def regTreeEval(tree, inputs):
            horizontal = len(inputs)
            xData = np.mat(np.ones((1, horizontal+1)))
            xData[0, 1:horizontal+1] = inputs
            ret = xData * tree.Value
            return float(ret)
        def modelTreeEval(tree, inputs):
            horizontal = len(inputs)
            inputsVar = np.mat(np.ones((1, horizontal + 1)))
            inputsVar[:, 1:horizontal+1] = inputs
            return float(np.mat(np.repeat(tree.Value, horizontal + 1)) * inputsVar.transpose())
        def treePredict(tree, inputs):
            if tree.Index is None:
                return regTreeEval(tree, inputs)
            if inputs[tree.Index] > tree.Value:
                if tree.leftNode.Index is not None:
                    return treePredict(tree.leftNode, inputs)
                else: return regTreeEval(tree.leftNode, inputs)
            else:
                if tree.rightNode.Index is not None:
                    return treePredict(tree.rightNode, inputs)
                else: return regTreeEval(tree.rightNode, inputs)
        return treePredict(self.Tree, inputs)
    def SmartTest(self, testData, testLabel, toggle_print = False, weight = None):
        if isinstance(testData, list) or isinstance(testData, tuple):
            testData = np.array(testData)
        if isinstance(testLabel, list) or isinstance(testLabel, tuple):
            testLabel = np.array(testLabel)
        assert testLabel.shape[0] == testData.shape[0]
        labelMean = np.mean(testLabel)
        predicts = list()
        for line in testData:
            predicts.append(self.Predict(line, weight))
        res, tot = float(), float()
        for i in range(len(predicts)):
            if toggle_print:
                print("predict: %.2f\treal: %.2f -> %.2f" % (predicts[i], testLabel[i], abs(predicts[i] - testLabel[i])))
            res += (testLabel[i] - predicts[i]) ** 2
            tot += (testLabel[i] - labelMean) ** 2
        self.r_square = (1 - res / float(tot))
        return self.r_square, np.array(predicts)
    def Graph(self, drawingFeature, labels = None, line = None):
        assert isinstance(labels, list) or isinstance(labels, tuple)
        if line is not None:
            predictedValue = np.array(line[-1])
            line = np.array(line[0])
        if isinstance(drawingFeature, list) or isinstance(drawingFeature, tuple):
            assert len(drawingFeature) in (1, 2)
            data = self.DataSet[:, drawingFeature]
        else:
            assert isinstance(drawingFeature, int)
            data = self.DataSet[:, [drawingFeature, ]]
        fig = plt.figure()
        if len(data[0]) == 1:
            graph = fig.add_subplot(111)
            graph.scatter(data, self.Labels, marker='o', c='#4E9BB9', s=100, alpha=0.2)
            if line is not None:
                graph.plot(line, predictedValue, color='#000000', lw=2)
            if labels is not None:
                assert len(labels) == 2
                graph.set_xlabel(labels[0])
                graph.set_ylabel(labels[1])
        else:
            graph = fig.add_subplot(111, projection='3d')
            graph.scatter(data[:, 0], data[:, 1], self.Labels, marker='o', c='#4E9BB9', s=20, alpha=0.1)
            if line is not None:
                x, y = np.meshgrid(line[:, 0], line[:, 1])
                graph.plot_wireframe(x, y, predictedValue[1:10], rstride=10, cstride=10)
            if labels is not None:
                assert len(labels) == 3
                graph.set_xlabel(labels[0])
                graph.set_ylabel(labels[1])
                graph.set_zlabel(labels[2])
        plt.show()
    def SaveGraph(self, name = None, path = None):
        if name is not None:
            assert isinstance(name, str)
        if path is not None:
            assert isinstance(path, str)
        try: plt.savefig(Util().GetDirectory() + "/DATA/save/" + path)
        except: print("Invalid Directory: %s" % Util().GetDirectory() + "/DATA/save/" + path)
        else: print("file saved to: %s" % Util().GetDirectory() + "/DATA/save/" + path)
