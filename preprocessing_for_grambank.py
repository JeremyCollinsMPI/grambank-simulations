from CreateDataFrame import *
from TreeFunctions import *
from copy import deepcopy
from simulation import *
import os
import json

def get_languages_in_grambank():
  data = readData('data.txt')
  # dict = createDictionary(df)
  languages = getUniqueLanguages(data)
  return languages

def get_grambank_value_dictionary():
  if 'grambank_value_dictionary.json' in os.listdir('.'):
    return json.load(open('grambank_value_dictionary.json', 'r'))
  df = readData('data.txt')
  grambank_value_dictionary = createDictionary(df) 
  json.dump(grambank_value_dictionary, open('grambank_value_dictionary.json', 'w'), indent=4)
  return grambank_value_dictionary

def further_preprocessing_of_grambank_value_dictionary(grambank_value_dictionary, feature_name):
  value_dictionary = {}
  for item in list(grambank_value_dictionary.items()):
    glottocode = item[0] 
    try:
      value = item[1][feature_name]
    except:
      value = None
    value_dictionary[glottocode] = value
  return value_dictionary
  
def make_input_and_output_arrays_for_grambank(value_dictionary, sample):
  input_array = make_input_array(value_dictionary)
  output_array = make_output_array(value_dictionary, sample)
  input_array = np.array([input_array])
  output_array = np.array([output_array])
  return input_array, output_array   

def make_na_array_1(value_dictionary):
  result = []
  sorted_keys = sorted(value_dictionary.keys())
  for key in sorted_keys:
    value = value_dictionary[key]
    if value == None:
      value = 0
    else:
      value = 1
    result.append(value)
  return np.array([[result]])

def make_na_array_2(value_dictionary, sample):
  result = []
  for item in sample:
    value = value_dictionary[item]
    if value == None:
      value = 0
    else:
      value = 1
    result.append(value)
  result = np.array(result)
  result = np.reshape(result, (np.shape(result)[0], 1))
  result = [result]
  result = np.array(result)
  result = np.array(result)
  '''
  should be of shape 1, samples, 1
  '''
  return result

def make_all_arrays_for_grambank(value_dictionary, trees, list_of_languages, sample, number_of_relatedness_bins=10, number_of_distance_bins=10):  
  input_array, output_array = make_input_and_output_arrays_for_grambank(value_dictionary, sample)
  na_array_1 = make_na_array_1(value_dictionary)
  na_array_2 = make_na_array_2(value_dictionary, sample)
  parent_dictionary = make_parent_dictionary(trees)
  relatedness_pairs_dictionary = make_relatedness_pairs_dictionary(list_of_languages, trees, parent_dictionary)
  distance_pairs_dictionary = make_distance_pairs_dictionary(list_of_languages)
  relatedness_array = make_relatedness_array(list_of_languages, sample, relatedness_pairs_dictionary)
  distance_array = make_distance_array(list_of_languages, sample, distance_pairs_dictionary)
  relatedness_array = preprocess_relatedness_array(relatedness_array, number_of_relatedness_bins)
  distance_array = preprocess_distance_array(distance_array, number_of_distance_bins)
  return input_array, output_array, relatedness_array, distance_array, na_array_1, na_array_2

    
'''
you need to make the na arrays at some point

'''