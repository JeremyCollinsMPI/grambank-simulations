from CreateDataFrame import *
import numpy as np
from numpy import random
from nexus import NexusReader



'''


relatedness_tensor of shape [samples, samples, 1]
relatedness_coefficients of shape [1, 1, 3]
you then want an array after doing ax^2 + bx + c
which is of shape [samples, samples, 1]
call this relatedness_tensor again.
then you are predicting features. 

the idea is that you are taking a random language,
finding the value for that feature,
then given the relatedness value, you are outputting a probability of having 1.
so you are just putting it through a sigmoid function.
you so far have a tensor relatedness_predictions of shape [samples, features]
then use cross-entropy against the data.















arrays that are needed;
the output array is the grambank data, so of shape [samples, features]
the input array is:

you have 

e.g.
the closest relative, which has the array of shape [1, features]
and a number showing how closely related it is, e.g. [1], if it is 1 node away,
and a number showing how geographically distant it is, e.g. [300].

so the input array is of shape [samples, samples, features]

there is a layer which then finds the values for features based on relatedness.
let's just do that first.

so,
a relatedness array, which is of shape [samples, samples, 1]
let's say this is numbers such as 1, 2, 3, 4...

this array is then put through some function is being learnt.  this is then the probability of 
the language having the same value of that feature.

so relatedness_probability is a number between 0 and 1.
so you then have a prediction, which is of shape [samples, features].
you find this by doing




'''




def rep(x,y):
	new=[]
	for m in range(y):
		new.append(x)
	return new	        


np.random.seed(10)

def make_grambank_dataframe(dataframe_given=False, df=None):
  data = readData('data.txt')
  # dict = createDictionary(df)
  languages = getUniqueLanguages(data)
  features = getUniqueFeatures(data)

  print(len(features))
  print(len(languages))
  if not dataframe_given:
    df = createDataFrame(data)
    samples = len(languages)
  else:
    samples = len(df)
  df = df.replace('?', np.nan) 
#   cheating at the moment by replacing '?' with 0.5
  df = df.replace(np.nan, 0.5)
  

  array = df.to_numpy()
  missing_data_matrix = np.ones(array.shape)

  print(array[0])
  return array, missing_data_matrix, samples, len(features), df


def find_relatedness(index1, index2, languages_dataframe):
  lineage1 = languages_dataframe.lineage[index1]
  lineage2 = languages_dataframe.lineage[index2]
  if pd.isnull(lineage1) or pd.isnull(lineage2):
    return 100
  lineage1 = lineage1.split('/')
  lineage2 = lineage2.split('/')
  genera_in_common = list(set(lineage1).intersection(set(lineage2)))
  if genera_in_common == []:
    return 100
  last_genus_in_common = genera_in_common[-1]
  position1 = len(lineage1) - lineage1.index(last_genus_in_common)
  position2 = len(lineage2) - lineage2.index(last_genus_in_common)
  return max(position1, position2)
  
def find_distance(index1, index2, languages_dataframe):
  lat1 = languages_dataframe.latitude[index1]
  lat2 = languages_dataframe.latitude[index2]
  lon1 = languages_dataframe.longitude[index1]
  lon2 = languages_dataframe.longitude[index2]
  if pd.isnull(lat1) or pd.isnull(lat2):
    print('OH NO')
    print(index1)
    print(index2)
  return haversine(lon1, lat1, lon2, lat2)

def make_relatedness_array(dataframe, languages_dataframe):
  result = []
  for index1 in dataframe.index:  
    temp = []
    for index2 in dataframe.index:
      print(index1)
      print(index2)
      if not index1 == index2:
        relatedness = find_relatedness(index1, index2, languages_dataframe)
        print(relatedness)
        temp.append(relatedness)
    result.append(temp)
  result = np.array(result)
  return result

def make_glottocode_pairs_array(dataframe, languages_dataframe):
  result = []
  for index1 in dataframe.index:  
    for index2 in dataframe.index:
      if not index1 == index2:
        result.append([index1, index2])
  result = np.array(result)
  return result

def make_distance_array(dataframe, languages_dataframe):
  result = []
  for index1 in dataframe.index:  
    temp = []
    for index2 in dataframe.index:
#       print(index1)
#       print(index2)
      if not index1 == index2:
        distance = find_distance(index1, index2, languages_dataframe)
#         print(distance)
        temp.append(distance)
    result.append(temp)
  result = np.array(result)
  return result




def make_simulated_array():
  '''
  structure should be (languages, samples)
  you have seven clusters
  e.g. ten features, 70 languages
  cluster1 = [1,1,1,0,0,0,0,0,1,1]
  cluster2 = [0,0,1,0,0,0,0,0,1,1]
  '''
  clusters = []
  number_of_clusters = 7
  features = 10
  languages_per_cluster = 10
  for i in range(number_of_clusters):
    clusters.append(random.randint(2,size =features))
  result = []
  for i in range(number_of_clusters):
    for j in range(languages_per_cluster):
      result.append(clusters[i])
  result = np.array(result)
  missing_data_matrix = np.ones(result.shape)
  input_array = []
  for i in range(number_of_clusters):
    to_append = rep(0, number_of_clusters)
    to_append[i] = 1
    for j in range(languages_per_cluster):
      input_array.append([to_append])      
  return result, missing_data_matrix, number_of_clusters*languages_per_cluster, features, input_array

def make_indo_european_array():
  n = NexusReader.from_file('IELex_Bouckaert2012.nex')
  df = pd.DataFrame.from_dict(n.data.matrix, orient='index')
  df = df.replace('?', np.nan) 
#   cheating at the moment by replacing '?' with 0
  df = df.replace(np.nan, 0)
  array = df.to_numpy()
  array = np.ndarray.astype(array, dtype=np.float32)
  missing_data_matrix = np.ones(array.shape, dtype=np.float32)
  number_of_clusters = 7
  features = 6280
  samples = 103
  return array, missing_data_matrix, samples, features, df


def iso_to_glottocode(iso, index=None, in_grambank=True):
  df = pd.read_csv('languages_and_dialects_geo.csv', header=0, index_col=0)
  df = df.dropna()
  glottocodes = df.index.where(df.isocodes==iso).dropna() 
  if len(glottocodes) == 0:
    return None
  return glottocodes[random.randint(len(glottocodes))]

# print(iso_to_glottocode('aau'))


def give_df_glottocode_indeces(df):
  ''' to do '''
  return df

# def make_phonotactics_array(languages_dataframe):
#   df = pd.read_csv('phonotactics.csv', header = 0, index_col=0)
#   df = give_df_glottocode_indeces(df)
#   ''' languages_dataframe should be given by main.py'''
#   phonotactics_relatedness_array = make_relatedness_array(df, languages_dataframe)
#   
#   ''' base it on this:'''
#   
#   df = df.replace('?', np.nan) 
# #   cheating at the moment by replacing '?' with 0
#   df = df.replace(np.nan, 0)
#   
# 
#   array = df.to_numpy()
#   missing_data_matrix = np.ones(array.shape)
# 
#   samples = 
#   features =   
#   
#   '''
#   columns f to fl inclusive
#   
#   
#   26 abcde
#   12 f
#   21 
#   
#   features = (26*5)+12+21
#   
#   features are from column 5 [base 0] to 167.
#   find the maximum value for each 
#   then divide the values by the maximum value
#   
#   you also want to give each row a (unique) glottocode
#   and then make that glottocode the index.
#   
#   languages_and_dialects_geo.csv
#   
#   
#   i also want to plan putting it together with the grambank data
#   
#   you can arbitrarily choose glottocodes; so you can have a function which 
#   returns a glottocode for an iso code.
#   you can also have it check for glottocodes that are in grambank. 
#   
#   what is the plan?
#   
#   you have the phonotactics dataframe.
#   you give them glottocodes.
#   if you are not aligning them with grambank data yet, then just the following;
#   you are making an array;
#   you are then taking the glottocodes in order and making a relatedness array for that dataframe.
#   
#   so you are using the function make_relatedness_array(dataframe, languages_dataframe)
#   this requires the indexes of language_dataframe to be glottocodes
#   
#   missing data array
#   
#   '''
#   
#   return array, missing_data_matrix, samples, features, df, phonotactics_relatedness_array









# def find_next_pair(index1, index2, last_index):
#   if index2 < last_index:
#     index2 = index2 + 1
#     return index1, index2
#   else:
#     if index1 < last_index - 1:
#       index1 = index1 + 1
#       index2 = index1 + 1
#       return index1, index2
#     else:
#       return None, None
# 
# def find_nulls(array1, array2):
#   result = array1 + array2
#   result = np.invert(np.isnan(result))
#   return result.astype(int)
# 
# def make_arrays():
#   input_array = []
#   output_array = []
#   null_layer = []
#   last_index = len(languages) - 1
#   index1 = 0
#   index2 = 1
#   end = False
#   while not end:
#     values1 = dict[languages[index1]]['values']
#     values2 = dict[languages[index2]]['values']
#     nulls = find_nulls(values1, values2)
#     input_array.append(values1)
#     output_array.append(values2)
#     null_layer.append(nulls)
#     index1, index2 = find_next_pair(index1, index2, last_index)
#     if index1 == None and index2 == None:
#       end = True
#       input_array = np.array(input_array)
#       output_array = np.array(output_array)
#       null_layer = np.array(null_layer)
#       input_array[np.isnan(input_array)] = 0
#       output_array[np.isnan(output_array)] = 0
#       return input_array, output_array, null_layer
# 
# def pickle_arrays():
#   input_array, output_array, null_layer = make_arrays()
#   np.save('input_array.npy', input_array)
#   np.save('output_array.npy', output_array)
#   np.save('null_layer.npy', null_layer)

# pickle_arrays()

# print(input_array[0])
# print(output_array[0])
# print(null_layer[0])
# print(np.shape(input_array))
# print(np.shape(output_array))
# print(np.shape(null_layer))


  
  