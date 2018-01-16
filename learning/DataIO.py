import numpy as np
from enum import Enum
import os
import re
from learning.Helper import Util


class DataIOError:

    def __init__(self):
        raise PermissionError("initializing class `DataIOError` is not allowed")

    def wrongTypeError(self, param, requiredType, curType = None):
        if curType is not None:
            raise TypeError("type for %s should be %s, but not %s"%(param, curType, requiredType))
        raise TypeError("type for %s should be %s"%(param, requiredType))

    def invalidFilePathError(self, path):
        if not isinstance(path, str):
            self.wrongTypeError(path.__name__, str, curType = type(path))
        raise FileNotFoundError("%s: file not found"%path)

    def fileFormatError(self, path):
        if not isinstance(path, str):
            self.wrongTypeError(path.__name__, str, curType=type(path))
        raise SyntaxError("file under %s has syntax issue (does not fit with the requirement"%path)


class AcceptedDataFormat(Enum):
    html = ".html"
    json = ".json"
    txt = ".txt"
    xml = ".xml"
    xls = ".xls"
    csv = ".csv"


class AcceptedLanguage(Enum):
    chinese = re.compile(r'([\u4E00-\u9FA5]+|\w+)')
    english = re.compile(r'[a-z|A-Z]+')


class DataIO:

    def __init__(self):
        self.dataset = None
        self.label = None
        self.title = None
        self.TYPE_DATASET_NDARRAY = np.array
        self.TYPE_DATASET_NDMAT = np.mat
        self.TYPE_DATASET_LIST = list
        self.__allowedDatatype = (self.TYPE_DATASET_LIST, self.TYPE_DATASET_NDMAT, self.TYPE_DATASET_NDARRAY)
        self.TYPE_VALUE_INT = int
        self.TYPE_VALUE_FLOAT = float
        self.TYPE_VALUE_STRING = str
        self.__allowedValuetype = (self.TYPE_VALUE_INT, self.TYPE_VALUE_FLOAT, self.TYPE_VALUE_STRING)

    def pathValidation(self, path):
        """
        DataIO accepts two style of input:
            1, an absolute path that leads to a file
            2, the name and file format
        if you choose the second style, the target file must be under learning/DATA/ folder
        :param path: str
        :return: None
        """
        if not isinstance(path, str):
            DataIOError.wrongTypeError(path.__name__, str, type(path))
        assert AcceptedDataFormat(path[-4::])
        if not os.path.exists(path):
            DataIOError.invalidFilePathError(path)
        print("file valid")


if __name__ == '__main__':
    demo = DataIO()
    demo.pathValidation("/User/Excited/.bath_profile")