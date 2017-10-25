import numpy as np
from Helper import DataPreprocessing
from Helper import Util

def estimate(dataSet, user, simMethod, item, toggle_print = False):
    horizontal = dataSet.shape[1]
    simTotal, ratSimTotal = float(), float()
    for i in range(horizontal):
        userRating = dataSet[user, i]
        if userRating == 0: continue
        overlap = np.nonzero(np.logical_and(np.array(dataSet[:, item]) > 0, np.array(dataSet[:, i]) > 0))[0]
        if len(overlap) == 0:
            similarity = 0
        else:
            similarity = simMethod(dataSet[overlap, item], dataSet[overlap, i])
            if toggle_print:
                print("similarity between (%d) and (%d) is %.6f%%"%(item, i, similarity * 100))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else: return ratSimTotal / simTotal

def recommend(dataSet, user, simMethod, prediction_num = 3, toggle_print = False):
    unratedItems = np.nonzero(np.array(dataSet[user, :]) == 0)[1]
    if len(unratedItems) == 0:
        print("no recommendation available")
    itemScores = list()
    for item in unratedItems:
        estimateScore = estimate(dataSet, user, simMethod, item, toggle_print)
        itemScores.append((item, estimateScore))
    return sorted(itemScores, key = lambda combination: combination[-1], reverse=True)[:prediction_num]

if __name__ == '__main__':
    demo = DataPreprocessing()
    demo.readSimpleDataSet("recommendation.txt", demo.ND_MAT, demo.INT, ", ")
    result = recommend(demo.DataSet, 2, Util().COSINE_SIM, toggle_print=True)
    print(result)