from __future__ import division
import pandas as pd
import numpy as np
from general import *
import math
  
def readData(file, index_col = 'Language_ID'):
	dataFrame = pd.read_csv(file, header = 0, sep =',', index_col = index_col)	
	return dataFrame

def getUniqueLanguages(dataframe):
  return sorted(unique(dataframe.index.tolist()))

def getUniqueFeatures(dataframe):
  return sorted(unique(dataframe['Parameter_ID']))

def getStates(dataframe, feature):
  filter = dataframe['Parameter_ID'].isin([feature])
  filtered = dataframe[filter]
  result = unique(filtered['Value'])
  result.remove('?')
  return result
  
def getMultistateFeatures(dataframe, uniqueFeatures = None):
  result = []
  if uniqueFeatures == None:
    uniqueFeatures = getUniqueFeatures(dataframe)
  for feature in uniqueFeatures:
    states = getStates(dataframe, feature)
    if len(states) > 2:
      result.append(feature)
  return result
  
def getValues(dataframe, language, feature):
  filtered = dataframe.loc[language]
  filter = filtered['Parameter_ID'].isin([feature])
  filtered = filtered[filter]
  x = filtered['Value']
  if x.empty:
    return '?'
  else:
    return x[0]

def getAllValues(dataframe, language, uniqueFeatures = None, multistateFeatures = None):
  if uniqueFeatures == None:
    uniqueFeatures = getUniqueFeatures(dataframe)
  values = []
  if multistateFeatures == None:
    multistateFeatures = getMultistateFeatures(dataframe, uniqueFeatures)
  for feature in uniqueFeatures:
    if not feature in multistateFeatures:
      value = getValues(dataframe, language, feature)
      values.append(value)
    else:
      for member in ['1','2']:
        if value == member or value == '3':
          values.append('1')
        elif value == '?':
          values.append('?')
        else:
          values.append('0')
  return values

def createDataFrame(dataframe):
  uniqueFeatures = getUniqueFeatures(dataframe)
  languages = getUniqueLanguages(dataframe)
  multistateFeatures = getMultistateFeatures(dataframe)
  rowsList = []
  for language in languages:
    print(language)
    values = getAllValues(dataframe, language, uniqueFeatures, multistateFeatures)
    rowsList.append(values)
  return pd.DataFrame(rowsList, index=languages)

def createDictionary(dataframe):
  multistateFeatures = getMultistateFeatures(dataframe)
  dict = {}
  for index, row in dataframe.iterrows():
    try:
      x = dict[index]
    except:
      dict[index] = {}
    if not row['Parameter_ID'] in multistateFeatures:
      dict[index][row['Parameter_ID']] = row['Value']
    else:
      value = row['Value']
      for member in ['1','2']:
        if value == member or value == '3':
          dict[index][row['Parameter_ID'] + '_' + member]  = '1'
        elif value == '?':
          dict[index][row['Parameter_ID'] + '_' + member]  = '?'
        else:
          dict[index][row['Parameter_ID'] + '_' + member]  = '0'
  features = getUniqueFeatures(dataframe)
  
  for language in dict.keys():
    values = []
    for feature in features:
      try:
        value = dict[language][feature]
        if value == '?':
          value = np.nan
        else:
          value = int(value)
      except:
        value = np.nan
      values.append(value)
    dict[language]['values'] = np.array(values)
  return dict

if __name__ == '__main__':
  df = readData('data.txt')
  features = getUniqueFeatures(df)
  f = getMultistateFeatures(df)
  print(f)
  x= getAllValues(df, 'zuni1245')
  print(x)
  x = createDataFrame(df)
  print(x)
  x = createDictionary(df)
  print(x['zuni1245']['values'])
