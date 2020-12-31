UNASSIGNED = 'Unassigned'
DO_NOT_DO = 'Do not do'

def addNode(tree, string):
	tree[string] = UNASSIGNED 
	if '(' in string and ')' in string:
		nodes = findChildren(string)
		for node in nodes:
			tree = addNode(tree, node)
	return tree

def createTree(string):
	tree = {}
	tree = addNode(tree, string)
	return tree
		
def renameNode(tree, nodeName, newName):
	outputTree = tree.copy()
	for otherNode in outputTree.keys():
		if ',' + nodeName + ':' in otherNode or '(' + nodeName + ':' in otherNode:
			del outputTree[otherNode]
			outputTree[otherNode.replace(nodeName, newName)] = UNASSIGNED 
		if findNodeName(otherNode) == nodeName:
			del outputTree[otherNode]
			outputTree[otherNode.replace(nodeName, newName)] = UNASSIGNED 
	return outputTree

def findBranchLength(node):
	branchLength = ''
	while not node == '':
		if not node[(len(node)-1)] == ';':
			branchLength = node[(len(node)-1)] + branchLength
		node = node[0:(len(node)-1)]
		if node[len(node)-1] == ':':
			return float(branchLength)
	return None

def findNodeName(node):
	branchLength = ''
	fullNodeName = node
	while not node[len(node)-1] == ')':
		branchLength = node[(len(node)-1)] + branchLength
		node = node[0:(len(node)-1)]
		if not node == '':
			if node[len(node)-1] == ':':
				return node[0:(len(node)-1)]
		if node == '':
			return fullNodeName
	return fullNodeName

def findStructure(node):
	branchLength = ''
	while not node == '':
		branchLength = node[(len(node)-1)] + branchLength
		node = node[0:(len(node)-1)]
		if not node == '':
			if node[len(node)-1] == ')':
				return node			
	return ''

def findChildren(node):
	if '(' in node and ')' in node:
		children = []
		node = findStructure(node)
		node = node[0:len(node)-1]
		node = node[1:len(node)]
		bracketsOpen = 0
		bracketsClosed = 0
		counter = 0
		string = ''
		while counter < len(node):
			if node[counter] == ',' and bracketsOpen == bracketsClosed:
				children.append(string)
				string = ''
			elif node[counter] == '(':
				bracketsOpen = bracketsOpen + 1
				string = string + '('
			elif node[counter] == ')':
				bracketsClosed = bracketsClosed + 1
				string = string + ')'
			else:
				string = string + node[counter]
			counter = counter + 1		
		children.append(string)
		return children
	return []		

def findParent(tree, nodeName):
	for otherNode in tree:
		children = findChildren(otherNode)
		if not children == None:
			for child in children:
				if findNodeName(child) == nodeName or child == nodeName:
					return otherNode
	return None

def findDescendantNodes(node):
	tree = createTree(node)
	nodes = tree.keys()
	return [x for x in nodes if not x == node]

def dropDescendantNodes(tree, node, newName):
	outputTree = tree.copy()
	nodesToDrop = findDescendantNodes(node)
	for nodeToDrop in nodesToDrop:
		del outputTree[nodeToDrop]
	nodeName = findNodeName(node)
	outputTree = renameNode(outputTree, nodeName, newName)
	return outputTree

def dropNode(tree, node):
	parent = findParent(tree, node)
	if parent == None:
		del tree[node]
		return tree
	children = findChildren(parent)
	children = [x for x in children if not x == node]
	if len(children) > 0:
		newName = '(' + ','.join(children) + ')'
		newName = newName + findNodeNameWithoutStructure(parent)
		tree = dropDescendantNodes(tree, parent, newName)
		for child in children:
			tree = addNode(tree, child)
	if len(children) == 0:
		newName = findNodeNameWithoutStructure(parent)
		tree = dropDescendantNodes(tree, parent, newName)
	return tree

def findDescendantTips(node):
	descendants = findDescendantNodes(node)
	tips = []
	for descendant in descendants:
		if not '(' in descendant and not ')' in descendant:
			tips.append(descendant)
	return tips

def findTips(tree):
	tips = []
	for node in tree.keys():
		if not '(' in node and not ')' in node:
			tips.append(node)
	return tips

def findNodeNameWithoutStructure(node):	
	nodeName = findNodeName(node)
	structure = findStructure(node)
	result = nodeName.replace(structure, '')
	return result

def findRoot(tree):
	for node in tree:
		if ';' in node:
			return node
	return None

def findAncestors(tree, node):
	results = []
	for node2 in tree:
		if not node2 == node and node in node2:
			results.append(node2)
	return results

def findDepth(tree, node):
	ancestors = findAncestors(tree, node)
	depth = 0
	for ancestor in ancestors:
		branchLength = float(findBranchLength(ancestor))
		depth = depth + branchLength
	return depth

def findMaximumHeight(node):
	children = findChildren(node)
	if children == []:
		return 0
	branchLength = float(findBranchLength(node))
	childrenMaximumHeights = []
	for child in children:
		maximumHeight = findMaximumHeight(child)
		branchLength = float(findBranchLength(child))
		childrenMaximumHeights.append(maximumHeight + branchLength)
	return max(childrenMaximumHeights)
	
def saveTreeToFile(tree, fileName = None):
	string = str(tree)
	if not fileName == None:
		file = open(fileName, 'w')
		file.write(string)
	else:
		fileName = findNodeNameWithoutStructure(findRoot(tree)) + '.txt'
		file = open(fileName, 'w')
		file.write(string)

def readTreeFromFile(fileName):
	file = open(fileName, 'r').read()
	tree = eval(file)
	return tree

def find_glottocode(node):
  '''
  example:
    'Itutang-Inapang [inap1241][mzu]-l-':1
  
  '''
  node_name = findNodeNameWithoutStructure(node)
  glottocode = node_name.split('[')[1].split(']')[0]
  return glottocode

def is_a_descendent(node_2, node_1):
  if node_2 in node_1:
    return True
  else:
    return False

def find_most_recent_common_ancestor(node_1, node_2, parent_dictionary):
  current_ancestor = node_1
  if is_a_descendent(node_2, current_ancestor):
    return current_ancestor
  if is_a_descendent(node_1, node_2):
    return node_2
  done = False
  while not done:
#     try:
#     print(current_ancestor)
#     print('here1')
    current_ancestor = parent_dictionary[current_ancestor]
#     except:
#       print('This should not happen')
#       return current_ancestor
    if is_a_descendent(node_2, current_ancestor):
      done = True
      return current_ancestor
    
def find_distance_from_ancestor_to_node(ancestor, node, parent_dictionary):
  distance = 0
  done = False
  current_ancestor = node
  if current_ancestor == ancestor:
    return 0
  while not done:
#     try:
    current_ancestor = parent_dictionary[current_ancestor]
#     except:
#       print('here2')
#       print(current_ancestor)
#       print('This should not happen')
#       return distance
    distance = distance + findBranchLength(current_ancestor)
    if current_ancestor == ancestor:
      return distance

def find_phyletic_distance(node_1, node_2, parent_dictionary):
  ancestor = find_most_recent_common_ancestor(node_1, node_2, parent_dictionary)
#   print(ancestor)
  distance_1 = find_distance_from_ancestor_to_node(ancestor, node_1, parent_dictionary)
  distance_2 = find_distance_from_ancestor_to_node(ancestor, node_2, parent_dictionary)
  return distance_1 + distance_2


# def label_nodes_that_do_not_have_descendents_in_a_list(tree, list_of_languages):
#   '''
#   
#   '''
