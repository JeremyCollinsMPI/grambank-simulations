from copy import deepcopy
import os
import json

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

def findBranchLength(node, string=False):
	branchLength = ''
	while not node == '':
		if not node[(len(node)-1)] == ';':
			branchLength = node[(len(node)-1)] + branchLength
		node = node[0:(len(node)-1)]
		if node[len(node)-1] == ':':
			if string:
				return branchLength
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

def find_ancestors(tree, node):
  ancestors = []
  finished = False
  current = node
  while not finished:
    current = findParent(tree, current)
    if current == None:
      finished = True
    else:
      ancestors.append(current)
  return ancestors

def change_names_of_node_and_ancestors(tree, node, new_name):
  ancestors = find_ancestors(tree, node)
  tree[new_name] = tree.pop(node)
  for ancestor in ancestors:
    new_ancestor_name = ancestor.replace(node, new_name)
    tree[new_ancestor_name] = tree.pop(ancestor)
#     print(new_ancestor_name)
#     print('&&&')
  return tree

def set_branch_length(node, branch_length):
  branchLength = ''
  to_add_at_the_end = ''
#   print('SETTING BRANCH LENGTH')
#   print(node)
  while not node == '':
#     print(node)
    if node[(len(node)-1)] == ';':
      to_add_at_the_end = ';'
    node = node[0:(len(node)-1)]
    if node[len(node)-1] == ':':
      node = node + str(branch_length) + to_add_at_the_end
      return node
  return None      

def make_node_into_parent(node, children):
  to_add_on_the_end = ''
  if node[len(node)-1] == ';':
    to_add_on_the_end = ';'
  branch_length = findBranchLength(node, string=True)
  node_name_without_structure = findNodeNameWithoutStructure(node)
  if len(children) == 0:
    return node_name_without_structure + ':' + branch_length + to_add_on_the_end
  to_return = '(' + ','.join(children) + ')' + node_name_without_structure + ':' + branch_length + to_add_on_the_end
  return to_return 

def get_rid_of_singleton_branches(tree):
  if tree == {}:
    return tree
  nodes = list(tree.keys())
  nodes_to_check = nodes
  finished = False  
  while not finished:
    node = nodes_to_check.pop()
    children = findChildren(node)
    if len(children) == 1:
      child = children[0]
      child_branch_length = findBranchLength(child)
      grandchildren = findChildren(child)
      del tree[child]
      branch_length = findBranchLength(node)
      new_node = make_node_into_parent(node, grandchildren)
      new_node = set_branch_length(new_node, branch_length + child_branch_length)
      change_names_of_node_and_ancestors(tree, node, new_node)
      nodes_to_check = list(tree.keys())
    if len(nodes_to_check) == 0:
      finished = True
  return tree
  
def drop_node_and_descendants(tree, node):
  if not node in tree:
    return tree
  descendants = findDescendantNodes(node)
  node_and_descendants = descendants + [node]
  for item in node_and_descendants:
    del tree[item]
  if tree == {}:
    return tree
  parent = findParent(tree, node)
  children = findChildren(parent)
  children = [x for x in children if not x == node]
  new_parent = make_node_into_parent(parent, children)
  tree = change_names_of_node_and_ancestors(tree, parent, new_parent)
  return tree

def node_not_in_list_and_descendants_not_in_list(node, list_of_languages):
  descendants = findDescendantNodes(node)
  node_and_descendants = descendants + [node]
  for item in node_and_descendants:
    glottocode = find_glottocode(item)
    if glottocode in list_of_languages:
      return False
  return True

def retain_only_nodes_that_are_in_list(tree, list_of_languages):
  nodes = list(tree.keys())
  for node in nodes:
    if not node in tree:
      continue
    if node_not_in_list_and_descendants_not_in_list(node, list_of_languages):
      print('removing')
      tree = drop_node_and_descendants(tree, node)
#   tree = get_rid_of_singleton_branches(tree)
  return tree

def make_reduced_trees(trees, list_of_languages, remake=False):
  if 'reduced_trees.json' in os.listdir('.') and not remake:
    return json.load(open('reduced_trees.json', 'r'))
  for i in range(len(trees)):
    tree = trees[i]
    tree = retain_only_nodes_that_are_in_list(tree, list_of_languages)
    trees[i] = tree
  json.dump(trees, open('reduced_trees.json', 'w'), indent=4)
  return trees



