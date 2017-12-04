import os
from learning.Helper import Util

#TODO: fix API and formats

class ItemSet:
    class combination(object):
        def __init__(self, newSet, newSupport, newFrequent):
            self.Set = newSet
            self.Support = newSupport
            self.Frequent = newFrequent
        def __repr__(self):
            setstring = " ".join([str(item) for item in self.Set])
            #return "set: (%s)\tsupport: %.2f%%"%(setstring, self.Support * 100)
            return setstring
    def __init__(self):
        self.Set = list()
    def push(self, newSet = None, newSupport = -1.0, newFrequent = 0):
        self.Set.append(self.combination(newSet, newSupport, newFrequent))
    def pop(self, index):
        self.Set.pop(index)
    def contain(self, target):
        for item in self.Set:
            if item.Set == target:
                return True
        return False
    def addFrequent(self, target):
        for item in self.Set:
            if item.Set == target:
                item.Frequent += 1
    def isEmpty(self):
        return len(self.Set) == 0
    def getSupport(self, target):
        for item in self.Set:
            if item.Set == target:
                return item.Support
    def find(self, target):
        for item in self.Set:
            if item.Set == target:
                return item

class Apriori:
    def __init__(self):
        self.DataSet = None
        self.AllElements = None
        self.SupportData = ItemSet()
        self.MIN_SUPPORT = 0.5
        self.MIN_CONFIDENCE = 0.7
    def ReadSimpleFile(self, path):
        assert isinstance(path, str)
        if path[-4::] != ".txt":
            print("Read file only support txt format")
            return None
        if not os.path.exists(Util().getDirectory() + "/DATA/" + path):
            print('File does not exist: %s' % (path))
            return None
        file = open(Util().getDirectory() + "/DATA/" + path, 'r')
        lines = file.readlines()
        self.DataSet = list()
        for line in lines:
            tempLine = line.strip().split(" ")
            tempLine = [int(item) for item in tempLine]
            self.DataSet.append(tempLine.copy())
    def __calcAllElements(self):
        assert self.DataSet is not None
        self.AllElements = set()
        for line in self.DataSet:
            self.AllElements = self.AllElements | set(line)
    def __initCombinations(self, data, combinations):
        validSet = list()
        itemSet = ItemSet()
        for line in data:
            for combination in combinations:
                if not isinstance(combination, set):
                    combination = {combination}
                if combination.issubset(line):
                    if not itemSet.contain(combination):
                        itemSet.push(combination)
                    itemSet.addFrequent(combination)
        vertical = float(len(data))
        for item in itemSet.Set:
            item.Support = item.Frequent / vertical
            if item.Support >= self.MIN_SUPPORT:
                validSet.append(item)
            if not self.SupportData.contain(item.Set):
                self.SupportData.push(item.Set, item.Support, item.Frequent)
        return validSet
    def __generateNextItemSet(self, previous, length):
        validSet = list()
        len_pre = len(previous)
        for i in range(len_pre):
            for j in range(i + 1, len_pre):
                if isinstance(previous[0], set):
                    left = list(previous[i])[:length - 2].sort()
                    right = list(previous[j])[:length - 2].sort()
                    if left == right:
                        newSet = (previous[i] | previous[j])
                else:
                    left = list(previous[i].Set)[:length - 2].sort()
                    right = list(previous[j].Set)[:length - 2].sort()
                    if left == right:
                        newSet = (previous[i].Set | previous[j].Set)
                if not newSet in validSet:
                    validSet.append(newSet)
        return validSet
    def generateValidItemSets(self):
        assert self.MIN_SUPPORT > 0.0 and self.MIN_SUPPORT < 1.0
        self.__calcAllElements()
        data = [set(item) for item in self.DataSet]
        validSet = self.__initCombinations(data, self.AllElements)
        validItemSet = list()
        validItemSet.append(validSet)
        index = 2
        while(len(validItemSet[index-2]) > 0):
            newSet = self.__generateNextItemSet(validItemSet[index-2], index)
            newValidSet = self.__initCombinations(data, newSet)
            validItemSet.append(newValidSet)
            index += 1
        return validItemSet
    def generateRules(self, itemSets, toggle_print=False):
        def calcConfidence(new_children, _parent):
            prune = list()
            for child in new_children:
                other = self.SupportData.find(parent.Set - child)
                if other is None:
                    return prune
                confidence = parent.Support / other.Support
                if confidence >= self.MIN_CONFIDENCE:
                    if toggle_print:
                        print("%s -> %s\tconfidence: %.2f%%" % (
                            other, self.SupportData.getSupport(child), confidence))
                ruleList.append((other, child, confidence))
                prune.append(child)
            return prune
        def ruleFromChild(_parent, _children):
            len_first_child = len(_children[0])
            if (len(parent.Set) > (len_first_child)):
                nextSet = self.__generateNextItemSet(_children, len_first_child + 1)
                nextSet = calcConfidence(nextSet, _parent)
                if(len(nextSet) > 1):
                    ruleFromChild(_parent, nextSet)
        ruleList = list()
        for i in range(1, len(itemSets)):
            for parent in itemSets[i]:
                children = [{item} for item in parent.Set]
                if i > 1: ruleFromChild(parent, children)
                else:
                    for child in children:
                        other = self.SupportData.find(parent.Set - child)
                        confidence = parent.Support / other.Support
                        if confidence >= self.MIN_CONFIDENCE:
                            if toggle_print:
                                print("%s -> %s\tconfidence: %.2f%%" % (other, child, confidence*100))
                        ruleList.append((other, child, confidence))
        return ruleList