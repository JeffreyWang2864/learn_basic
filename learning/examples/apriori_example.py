from learning.machinelearning.association import Apriori

demo = Apriori()
demo.ReadSimpleFile("association_test.txt")
print(demo.DataSet)
demo.MIN_SUPPORT = 0.5
demo.MIN_CONFIDENCE = 0.7
vis = demo.generateValidItemSets()
result = demo.generateRules(vis, True)
print(result)

print("------------------------------------------")

demo = Apriori()
demo.ReadSimpleFile("aprioriDataSet02.txt")
demo.MIN_SUPPORT = 0.03
demo.MIN_CONFIDENCE = 0.3
vis = demo.generateValidItemSets()
result = demo.generateRules(vis, False)
print(result)