from TreeFunctions import *
from ConvertToWalsTree import *

trees = open('trees.txt','r').readlines()


# tree = createTree(treeString)
# tips = findDescendantTips(treeString)
# for tip in tips:
# 	print tip
# 	print hasIsoCode(tip)

def allTipsHaveIsoCodesTest():
	for treeString in trees:
		tree = createTree(treeString)
		print treeString
		print allTipsHaveIsoCodes(tree)
 



def ensureAllTipsHaveIsoCodesTest():

	for treeString in trees:
		tree = createTree(treeString)
# 		print treeString
		x = ensureAllTipsHaveIsoCodes(tree)
# 		print x
		print allTipsHaveIsoCodes(x)
ensureAllTipsHaveIsoCodesTest()

