from __future__ import division
import pandas
from TreeFunctions import *
from PrepareWalsData import *
import numpy as np
from scipy.linalg import fractional_matrix_power
from general import *
from ConvertToWalsTree import * 
from math import sin, cos, asin, sqrt

def makeStateMatrix(states):
	matrix = []
	for state in states:
		toAppend = []
		for state in states:
			toAppend.append(0)
		matrix.append(toAppend)
	return matrix

def assignTipValuesByIso(tree, dataFrame, featureName):
	states = findStates(dataFrame, featureName)
	outputTree = tree.copy()
	tips = findTips(outputTree)
	for tip in tips:
		outputTree[tip] = {}
		iso = findIsoCode(tip)
		value = lookUpValueForIso(iso, dataFrame, featureName)
		outputTree[tip]['states'] = {}
		if value == '?':
			for state in states:
				outputTree[tip]['states'][state] = '?'
		else:
			for state in states:
				if state == value:
					outputTree[tip]['states'][state] = 1
				else:
					outputTree[tip]['states'][state] = 0
	return outputTree
				
def branchLengthsAreAllOne(tree):
	result = True
	for node in tree:
		branchLength = findBranchLength(node)
		if branchLength == '1':
			pass
		else:
			return False

def findTransitionProbability(state1, state2, states, matrix, branchLength):
	state1_index = states.index(state1)
	state2_index = states.index(state2)
	matrixRaisedToPowerOfBranchLength = fractional_matrix_power(matrix, branchLength)
	return matrixRaisedToPowerOfBranchLength[state1_index, state2_index]
			
def calculateLikelihoodForNode(inputTree, node, states, matrix):
	tree = inputTree.copy()
	children = findChildren(node)
	for child in children:
		if tree[child] == 'Unassigned':
			tree[node] = 'Unassigned'
			return tree, False
	tree[node] = {}
	for child in children:
		branchLength = float(findBranchLength(child))
		tree[node][child] = {}
		for state1 in states:
			tree[node][child][state1]  = {}
			for state2 in states:
				likelihood = tree[child]['states'][state2]
				if likelihood == '?':
					tree[node][child][state1][state2] = '?'
				else:
					tree[node][child][state1][state2] = likelihood * findTransitionProbability(state1, state2, states, matrix, branchLength)
	tree[node]['states'] = {}
	for state1 in states:
		total = 1
		sub_totals = []
		for child in children:				
			sub_total = 0
			for state2 in states:
				likelihood = tree[node][child][state1][state2]
				if likelihood == '?':
					sub_total = '?'
				else:
# 					print sub_total
# 					print likelihood
					sub_total = sub_total + likelihood
			sub_totals.append(sub_total)
		sub_totals = [x for x in sub_totals if not x == '?']
		if len(sub_totals) == 0:
			total = '?'
		else:
			total = np.prod(sub_totals)
		tree[node]['states'][state1] = total
	return tree, True

def calculateLikelihoodForAllNodes(inputTree, states, matrix):
	tree = inputTree.copy()
	done = False
	while done == False:
		done = True
		for node in tree:
			if tree[node] == 'Unassigned':
				tree, nodeDone = calculateLikelihoodForNode(tree, node, states, matrix)
# 				print node
# 				print tree[node]
				if not nodeDone:
					done = False
	return tree

def findLikelihood(inputTree, states, matrix):
	tree = calculateLikelihoodForAllNodes(inputTree, states, matrix)
	root = findRoot(tree)
	total = 0
	for state in states:
		total = total + tree[root]['states'][state]
	total = total/len(states)
	return total
	





	