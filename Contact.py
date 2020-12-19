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
import random

def assignCoordinatesByIso(tree, dataFrame, latitudeName, longitudeName):
	outputTree = tree.copy()
	tips = findTips(outputTree)
	for tip in tips:
		outputTree[tip] = {}
		iso = findIsoCode(tip)
		latitude = lookUpValueForIso(iso, dataFrame, latitudeName)
		longitude = lookUpValueForIso(iso, dataFrame, longitudeName)
		try:
			outputTree[tip]['latitude'] = float(latitude)
		except:
			outputTree[tip]['latitude'] = latitude
		try:
			outputTree[tip]['longitude'] = float(longitude)
		except:
			outputTree[tip]['longitude'] = longitude
	return outputTree	
	
def reconstructLocationForNode(inputTree, node):
	tree = inputTree.copy()
	children = findChildren(node)
	for child in children:
		if tree[child] == 'Unassigned':
			tree[node] = 'Unassigned'
			return tree, False
	tree[node] = {}
	latitudes = []
	longitudes = []
	for child in children:
		latitudes.append(tree[child]['latitude'])
		longitudes.append(tree[child]['longitude'])
	latitudes = [x for x in latitudes if not x == '?']
	longitudes = [x for x in longitudes if not x == '?']
	if len(latitudes) == 0:
		tree[node]['latitude'] = '?'
	else:
		tree[node]['latitude'] = np.mean(latitudes)
	if len(longitudes) == 0:
		tree[node]['longitude'] = '?'
	else:
		tree[node]['longitude'] = np.mean(longitudes)
	return tree, True

def reconstructLocationsForAllNodes(inputTree):
	tree = inputTree.copy()
	done = False
	while done == False:
		done = True
		for node in tree:
			if tree[node] == 'Unassigned':
				tree, nodeDone = reconstructLocationForNode(tree, node)
				if not nodeDone:
					done = False
	return tree

def findLongitude(iso, dataFrame):
	return lookUpValueForIso(iso, dataFrame, 'longitude')

def findLatitude(iso, dataFrame):
	return lookUpValueForIso(iso, dataFrame, 'latitude')
				
def findProbabilityOfState(state, states):
	if len(states) > 0:
		return (states.count(state))/len(states)
	else:
		return 0



def findNodesWithinACertainDistance(nodes, latitude, longitude, distanceThreshold, limit, nodesToInclude = [], useAllNodes = False):
	result = []
	lon1 = longitude
	lat1 = latitude
	for node in nodes:
		if useAllNodes or node in nodesToInclude:
			if not nodes[node]['latitude'] == '?' and not nodes[node]['longitude'] == '?':
				lon2 = nodes[node]['longitude']
				lat2 = nodes[node]['latitude']
				if not limit == None:
					if abs(lat1 - lat2) < limit and abs(lon1 - lon2) < limit:
						if distanceThreshold == None:
							result.append(node)
						else:
							distance = haversine(lon1, lat1, lon2, lat2)
							if distance < distanceThreshold:
								result.append(node)	
				else:
					distance = haversine(lon1, lat1, lon2, lat2)
					if distance < distanceThreshold:
						result.append(node)
	return result							


def findStatesFromNodesWithinACertainDistance(reconstructedStates, nodes, latitude, longitude, distanceThreshold, limit, nodesToInclude = [], useAllNodes = False):
	result = []
	nodesNearby = newFindNodesWithinACertainDistance(nodes, latitude, longitude, distanceThreshold, limit, nodesToInclude, useAllNodes)
	for node in nodesNearby:
		if node in reconstructedStates:
			statesToReturn = reconstructedStates[node]
			result.append(statesToReturn)
		else:
			pass
# 			print 'Node is not in reconstructed states.'
	return result

# def findNodesToInclude(node, nodeHeightsList, nodeHeightsDictionary):
# 	
# 


def newFindNodesWithinACertainDistance(nodes, latitude, longitude, distanceThreshold, limit, nodesToInclude = {}, useAllNodes = False):
	result = []
	lon1 = longitude
	lat1 = latitude
	for node in nodes:
		if useAllNodes or inDict(node, nodesToInclude):
			if not nodes[node]['latitude'] == '?' and not nodes[node]['longitude'] == '?':
				lon2 = nodes[node]['longitude']
				lat2 = nodes[node]['latitude']
				if not limit == None:
					if abs(lat1 - lat2) < limit and abs(lon1 - lon2) < limit:					
						if distanceThreshold == None:
							if checkNodeIsMostRecent:		
								result.append(node)
						else:
							distance = haversine(lon1, lat1, lon2, lat2)
							if distance < distanceThreshold:
								if checkNodeIsMostRecent:		
									result.append(node)
				else:
					distance = haversine(lon1, lat1, lon2, lat2)
					if distance < distanceThreshold:
						if checkNodeIsMostRecent:		
							result.append(node)
	return result	

def newFindStatesFromNodesWithinACertainDistance(reconstructedStates, nodes, latitude, longitude, distanceThreshold, limit, nodesToInclude = {}, useAllNodes = False):
	result = []
	nodesNearby = newFindNodesWithinACertainDistance(nodes, latitude, longitude, distanceThreshold, limit, nodesToInclude, useAllNodes)
	for node in nodesNearby:
		if node in reconstructedStates:
			statesToReturn = reconstructedStates[node]
			result.append(statesToReturn)
		else:
			pass
	return result

# currently working on this
def newFindContactStatesForNode(inputTree, node, states, reconstructedStates, nodesLocations, nodeHeightsList, nodeHeightsDictionary, distanceThreshold, limit):
	tree = inputTree.copy()
	result = {}
	location = nodesLocations[node]
	latitude = location['latitude']
	longitude = location['longitude']
	if longitude == '?' or latitude == '?':
		for state in states:
			result[state] = '?'
	else:
		nodesToInclude = findNodesToInclude(node, temporalOrder)
		statesToSumOver = newFindStatesFromNodesWithinACertainDistance(reconstructedStates, nodesLocations, latitude, longitude, distanceThreshold, limit, nodesToInclude)
# 		print 'moose'
# 		print statesToSumOver
		for state in states:
			total = 0
			for stateToSumOver in statesToSumOver:
				print stateToSumOver
				total = total + stateToSumOver[state]
			if len(statesToSumOver) > 0:
				total = total / len(statesToSumOver)
			result[state] = total
	tree[node] = result
# 	print 'am'
# 	print result
	return tree
	
def newFindContactStatesForAllNodes(inputTree, states, reconstructedStates, nodesLocations, temporalOrder, distanceThreshold, limit):
	tree = inputTree.copy()
	for node in tree:
		tree[node] = 'Unassigned'
	i = 0
	for node in tree:
# 		print node
		tree = newFindContactStatesForNode(tree, node, states, reconstructedStates, nodesLocations, temporalOrder, distanceThreshold, limit)
		i = i +1
# 		print i
	return tree 


	

	
