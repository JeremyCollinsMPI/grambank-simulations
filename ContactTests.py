from __future__ import division
import pandas
from TreeFunctions import *
from PrepareWalsData import *
import numpy as np
from scipy.linalg import fractional_matrix_power
from general import *
from ConvertToWalsTree import * 
from math import sin, cos, asin, sqrt
from LikelihoodFunction import *
from Contact import *

trees = open('trees.txt').readlines()


dataFrame = assignIsoValues('language.csv')


# numeral noun
# featureName = '89A Order of Numeral and Noun'
# dataFrame = filterDataFrame(dataFrame, featureName, ['2 Noun-Numeral', '1 Numeral-Noun'])
# states = findStates(dataFrame, featureName)
# print states

#  subject verb
# featureName = '82A Order of Subject and Verb'
# dataFrame = filterDataFrame(dataFrame, featureName, ['1 SV', '2 VS'])
# states = findStates(dataFrame, featureName)
# print states

# object-verb
featureName = '83A Order of Object and Verb'
dataFrame = filterDataFrame(dataFrame, featureName, ['1 OV', '2 VO'])
states = findStates(dataFrame, featureName)
print states
 
tree = trees[66]
tree = tree.strip('\n')


tree = createTree(tree)
tree = ensureAllTipsHaveIsoCodes(tree)

def statesWithinDistanceTest():
	latitude = 20.25
	longitude = 83.16666666665
	statesWithinDistance = findStatesWithinDistance(latitude, longitude, dataFrame, featureName, distanceThreshold = None, limit = 10, isosToExclude = [])
	print statesWithinDistance
	return statesWithinDistance

def contactStatesTreeTest():
	locationTree = assignCoordinatesByIso(tree, dataFrame, 'latitude', 'longitude')
	locationTree = reconstructLocationsForAllNodes(locationTree)
	for node in locationTree:
		print findNodeNameWithoutStructure(node)
		print locationTree[node]

	distanceThreshold = None
	limit = 10
	contactStatesTree = findContactStatesForAllNodes(tree, locationTree, dataFrame, featureName, distanceThreshold, limit)
	for node in contactStatesTree:
		print node
		print contactStatesTree[node]
	print contactStatesTree
	saveTreeToFile(contactStatesTree)


def findProbabilityOfStateTest():
	states = statesWithinDistanceTest()
	print findProbabilityOfState('1 Numeral-Noun', states)


contactStatesTreeTest()



# findProbabilityOfStateTest()
# contactStateTree = readTreeFromFile("'Austroasiatic [aust1305]'.txt")
# print contactStateTree
