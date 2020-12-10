from TreeFunctions import *

trees = open('trees.txt','r').readlines()

def renameNodeTest():
	treeString = "(('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1,'Yagua [yagu1244][yad]-l-':1)'Peba-Yagua [peba1241]':1;"
	tree = createTree(treeString)
	test1 = renameNode(tree, "'Peba [peba1243]-l-'", "moose")
	test2 = renameNode(tree,  "('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]'", "moose")

def findNodeNameTest():
	print findNodeName("('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1")
	print findNodeName("moose")
	print findNodeName("(moose, poppet)arnold")
	print findNodeName("(moose:1, poppet:1)arnold")
	print findNodeName("(moose:1, poppet:1)arnold:1")

def findBranchLengthTest():	
	print findBranchLength("('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1")

def findStructureTest():
	print findStructure("('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1")

def findChildrenTest():
	print findChildren("('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1")	
	print findChildren("(('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1,'Yagua [yagu1244][yad]-l-':1)'Peba-Yagua [peba1241]':1;")
	print findChildren("'Peba [peba1243]-l-':1")
	print findChildren("(('Amux [amux1234]':1)'Southwestern Dargwa [sout3261]-l-':1)'South Dargwa [sout3260]':1")

def findParentTest():
	treeString = "(('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1,'Yagua [yagu1244][yad]-l-':1)'Peba-Yagua [peba1241]':1;"
	tree = {}
	tree = addNode(tree, treeString)
	test1 = (findParent(tree, "'Yameo [yame1242][yme]-l-'") == "('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1")
	test2 = (findParent(tree, "('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1") == "(('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1,'Yagua [yagu1244][yad]-l-':1)'Peba-Yagua [peba1241]':1;")
	test3 = (findParent(tree, "'Peba [peba1243]-l-':1") == "('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1")
	print test1, test2, test3

def findDescendantNodesTest():
	treeString = "(('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1,'Yagua [yagu1244][yad]-l-':1)'Peba-Yagua [peba1241]':1;"
	print findDescendantNodes(treeString)	

def dropDescendantNodesTest():
	treeString = "(('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1,'Yagua [yagu1244][yad]-l-':1)'Peba-Yagua [peba1241]':1;"
	tree = createTree(treeString)
	tree = dropDescendantNodes(tree, "('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1", 'dog:1')
	print tree

def dropNodeTest():
	treeString = "(('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1,'Yagua [yagu1244][yad]-l-':1)'Peba-Yagua [peba1241]':1;"
	tree = createTree(treeString)
	test1 = (dropNode(tree, "('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1") == {"'Yagua [yagu1244][yad]-l-':1": 'NA', "('Yagua [yagu1244][yad]-l-':1)'Peba-Yagua [peba1241]':1;": 'NA'})
	test2 = dropNode(tree, "'Peba [peba1243]-l-':1")
	print dropNode(tree, "('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1") 
	print test1
	print {"'Yagua [yagu1244][yad]-l-':1": 'NA', "('Yagua [yagu1244][yad]-l-':1)Peba-Yagua [peba1241]':1;": 'NA'}

def findDescendantTipsTest():
	treeString = "(('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1,'Yagua [yagu1244][yad]-l-':1)'Peba-Yagua [peba1241]':1;"
	test1 = (findDescendantTips("'Peba [peba1243]-l-':1") == [])
	test2 = (findDescendantTips("('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1") == ["'Peba [peba1243]-l-':1", "'Yameo [yame1242][yme]-l-':1"])
	test3 = (findDescendantTips(treeString) == ["'Yagua [yagu1244][yad]-l-':1", "'Peba [peba1243]-l-':1", "'Yameo [yame1242][yme]-l-':1"])
	print test1, test2, test3

def findNodeNameWithoutStructureTest():
	print findNodeNameWithoutStructure("('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1")
	print findNodeNameWithoutStructure("moose")
	print findNodeNameWithoutStructure("(moose, poppet)arnold")
	print findNodeNameWithoutStructure("(moose:1, poppet:1)arnold")
	print findNodeNameWithoutStructure("(moose:1, poppet:1)arnold:1")

def createTreeTest():
	tree = createTree("'Kubachi [kuba1248]-l-':1,'North-Central Dargwa [darg1241][dar]-l-':1,(('Amux [amux1234]':1)'Southwestern Dargwa [sout3261]-l-':1)'South Dargwa [sout3260]':1)'Dargwic [darg1242]':1;")
	print tree

def findAncestorsTest():
	treeString = "(('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1,'Yagua [yagu1244][yad]-l-':1)'Peba-Yagua [peba1241]':1;"
	tree = createTree(treeString)
	print findAncestors(tree, "'Peba [peba1243]-l-':1")

def findDepthTest():
	treeString = "(('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1,'Yagua [yagu1244][yad]-l-':1)'Peba-Yagua [peba1241]':1;"
	tree = createTree(treeString)
	print findDepth(tree, "'Peba [peba1243]-l-':1")

def findMaximumHeightTest():
	node = "(('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1,'Yagua [yagu1244][yad]-l-':1)'Peba-Yagua [peba1241]':1;"
# 	node = "('Peba [peba1243]-l-':1,'Yameo [yame1242][yme]-l-':1)'Peba-Yameo [peba1242]':1;"
# 	node = "'Peba [peba1243]-l-':1;"
	print findMaximumHeight(node)

findMaximumHeightTest()