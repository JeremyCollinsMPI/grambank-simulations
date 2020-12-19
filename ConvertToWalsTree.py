from TreeFunctions import *



def hasIsoCode(node):		
	# should look up iso code list
	nodeName = findNodeNameWithoutStructure(node)
	try:
		isoCode = nodeName.split('[')[2].split(']')[0]
		return True
	except:
		return False

def findIsoCode(node):	
	nodeName = findNodeNameWithoutStructure(node)
	try:
		isoCode = nodeName.split('[')[2].split(']')[0]
		return isoCode
	except:
		return None	

def allTipsHaveIsoCodes(tree):
	tips = findTips(tree)
	result = True
	for tip in tips:
		if not hasIsoCode(tip):
			result = False
	return result

def ensureAllTipsHaveIsoCodes(tree):
	nodes = tree.keys()
	done = False
	while done == False:
		restart = False
		nodes = tree.keys()
		if nodes == []:
			done = True
		tips = findTips(tree)
		for node in nodes:
			if hasIsoCode(node):
				isoCode = findIsoCode(node)
				descendants = findDescendantNodes(node)
				if len(descendants) > 0:
					newName = findNodeNameWithoutStructure(node)
					tree = dropDescendantNodes(tree, node, newName)
					restart = True
					break
			else:
				if node in tips:
					tree = dropNode(tree, node)
					restart = True
					break
		if not restart:
			done = True
	return tree


			
		

			