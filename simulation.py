from scipy.linalg import fractional_matrix_power
from TreeFunctions import *
import numpy as np
from general import *
from copy import deepcopy
import multiprocessing
import os
import json
import pandas as pd

dir = os.listdir('.')

def timeit(method):
  def timed(*args, **kw):
    ts = time.time()
    result = method(*args, **kw)
    te = time.time()
    if 'log_time' in kw:
      name = kw.get('log_name', method.__name__.upper())
      kw['log_time'][name] = int((te - ts))
    else:
      print('%r  %2.2f s' % \
      (method.__name__, (te - ts) ))
    return result
  return timed 

def find_parent(node, parent_dictionary):
  try:
    parent = parent_dictionary[node]
    return parent
  except:
    return None

def prepare_node_name(node):
  return node

def make_trees():
  trees = open('trees.txt','r').readlines()
  for i in range(len(trees)):
    tree = trees[i]
    tree = tree.strip('\n')
    tree = createTree(tree)
    trees[i] = tree
  return trees

def make_nodes_to_tree_dictionary(trees):
  if 'nodes_to_tree_dictionary.json' in dir:
    return json.load(open('nodes_to_tree_dictionary.json', 'r'))
  nodes_to_tree_dictionary = {}
  for i in range(len(trees)):
    tree = trees[i]
    for key in tree.keys():
      node_name = prepare_node_name(key)
      nodes_to_tree_dictionary[node_name] = i
  json.dump(nodes_to_tree_dictionary, open('nodes_to_tree_dictionary.json', 'w'), indent=4)
  return nodes_to_tree_dictionary

def get_locations(trees):
  if 'locations.json' in dir:
    return json.load(open('locations.json', 'r'))  
  df = pd.read_csv('Languages.csv', index_col='id')
  df2 = pd.read_csv('languages_and_dialects_geo.csv', index_col='glottocode')
  locations = {}
  for tree in trees:
    for key in tree.keys():
      glottocode = find_glottocode(key)
      nodename = prepare_node_name(key)
      if glottocode in df.index:
        lat = df['latitude'][glottocode]
        long = df['longitude'][glottocode]
        if not lat == None and not long == None:
          locations[nodename] = {'lat': lat, 'long': long}
      else:
        if glottocode in df2.index:
          lat = df2['latitude'][glottocode]
          long = df2['longitude'][glottocode]
          if not np.isnan(lat) and not np.isnan(long): 
            locations[nodename] = {'lat': lat, 'long': long}
  json.dump(locations, open('locations.json', 'w'), indent=4)
  return locations

def process_longs_part_1(longs):
  for i in range(len(longs)):
    long = longs[i]
    if long < -30:
      long = long + 360
      longs[i] = long
  return longs

def process_longs_part_2(long):
  if long > 180:
    long = long - 360
  return long
    
def reconstruct_location_for_node(tree, node, reconstructed_locations_dictionary, locations):
  children = findChildren(node)
  for child in children:
    if reconstructed_locations_dictionary[child] == 'Unassigned':
      reconstructed_locations_dictionary[node] == 'Unassigned'
      node_done = False
      return node_done
  lats = []
  longs = []
  for child in children:
    location = reconstructed_locations_dictionary[child]
    if not location == None:
      lats.append(location['lat'])
      longs.append(location['long'])
  if len(lats) == 0:
    reconstructed_locations_dictionary[node] = None
  else:
    average_lat = np.mean(lats)
    longs = process_longs_part_1(longs)
    average_long = np.mean(longs)
    average_long = process_longs_part_2(average_long)
    reconstructed_locations_dictionary[node] = {'lat': average_lat, 'long': average_long}
  node_done = True
  return node_done
    
def reconstruct_locations_for_all_nodes(tree, reconstructed_locations_dictionary, locations):
  done = False
  while not done:
    done = True
    for node in tree:
      if reconstructed_locations_dictionary[node] == 'Unassigned':
        node_done = reconstruct_location_for_node(tree, node, reconstructed_locations_dictionary, locations)
        if not node_done:
          done = False
           
def make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary):
  if 'reconstructed_locations_dictionary.json' in dir:
    return json.load(open('reconstructed_locations_dictionary.json', 'r'))  
  reconstructed_locations_dictionary = {}
  nodes_to_check = []
  for tree in trees:
    for key in tree.keys():
      nodename = prepare_node_name(key) 
      reconstructed_locations_dictionary[nodename] = 'Unassigned'
      try:
        location = locations[nodename]
        reconstructed_locations_dictionary[nodename] = location
      except:
        pass    
  for i in range(len(trees)):
    tree = trees[i]
    reconstruct_locations_for_all_nodes(tree, reconstructed_locations_dictionary, locations)
  json.dump(reconstructed_locations_dictionary, open('reconstructed_locations_dictionary.json', 'w'), indent=4)
  return reconstructed_locations_dictionary

def add_node_to_time_depths_dictionary(node, time_depths_dictionary, current_time_depth):
  time_depths_dictionary[node] = current_time_depth
  children = findChildren(node)
  for child in children:
    branch_length = findBranchLength(child)
    new_time_depth = current_time_depth + branch_length
    add_node_to_time_depths_dictionary(child, time_depths_dictionary, new_time_depth)
  return time_depths_dictionary
  
def make_time_depths_dictionary(trees):
  if 'time_depths_dictionary.json' in dir:
    return json.load(open('time_depths_dictionary.json', 'r'))
  time_depths_dictionary = {}
  for tree in trees:
    root = findRoot(tree)
    root_time_depth = 0
    time_depths_dictionary = add_node_to_time_depths_dictionary(root, time_depths_dictionary, root_time_depth)
  json.dump(time_depths_dictionary, open('time_depths_dictionary.json', 'w'), indent=4)
  return time_depths_dictionary

def make_parent_dictionary(trees):
  if 'parent_dictionary.json' in dir:
    return json.load(open('parent_dictionary.json', 'r'))
  parent_dictionary = {}
  for tree in trees:
    for key in tree.keys():
      children = findChildren(key)
      for child in children:
        parent_dictionary[child] = key
  json.dump(parent_dictionary, open('parent_dictionary.json', 'w'), indent=4)
  return parent_dictionary

def neighbouring(A, B, reconstructed_locations_dictionary, distance_threshold=500, lat_limit=10, long_limit=10):
  A_location = reconstructed_locations_dictionary[A]
  B_location = reconstructed_locations_dictionary[B]
  if A_location == None or B_location == None:
    return False
  A_lat = A_location['lat']
  A_long = A_location['long']
  B_lat = B_location['lat']
  B_long = B_location['long']
  if abs(A_lat - B_lat) < lat_limit and abs(A_long-B_long) < long_limit:
    distance = haversine(A_long, A_lat, B_long, B_lat)
    if distance < distance_threshold:
      return True
    else:  
      return False
  else:
    return False

def is_eldest_node_younger_than_A(B, A, tree, time_depths_dictionary, A_time_depth ,B_time_depth, parent_dictionary):
  B_parent = find_parent(B, parent_dictionary)
  B_parent_time_depth = time_depths_dictionary[B_parent]
  if B_time_depth >= A_time_depth and B_parent_time_depth < A_time_depth:
    return True
  return False

def overlapping_range(min1, max1, min2, max2):
  if min2 <= max1 and min2 >= min1:
    return True
  else:
    return False

def correct_time_depth(A, B, tree, time_depths_dictionary, A_time_depth, parent_dictionary):
  A_beginning_time_depth = A_time_depth
  A_branch_length = findBranchLength(A)
  A_end_time_depth = A_beginning_time_depth + A_branch_length
  try:
    B_parent = parent_dictionary[B]
    B_parent_beginning_time_depth = time_depths_dictionary[B_parent]
    B_parent_branch_length = findBranchLength(B_parent)
    B_parent_end_time_depth = B_parent_beginning_time_depth + B_parent_branch_length
  except:
    B_parent_beginning_time_depth = 0
    B_parent_end_time_depth = 0 
  if overlapping_range(A_beginning_time_depth, A_end_time_depth, B_parent_beginning_time_depth, B_parent_end_time_depth):
    return True
  else:
    return False

def is_not_parent_or_child(A, B, parent_dictionary):
  try:
    if parent_dictionary[A] == B:
      return False
  except:
    pass
  try:
    if parent_dictionary[B] == A:
      return False
  except:
    pass
  return True

def find_contemporary_neighbours(node_A, trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary):
  contemporary_neighbours = []
  for i in range(len(trees)):
    tree = trees[i]
    for node_B in tree.keys():
      if not node_A == node_B and is_not_parent_or_child(node_A, node_B, parent_dictionary):
        A_time_depth = time_depths_dictionary[node_A]
        if correct_time_depth(node_A, node_B, tree, time_depths_dictionary, A_time_depth, parent_dictionary):
          if neighbouring(node_A, node_B, reconstructed_locations_dictionary):
            time_depth = time_depths_dictionary[node_B]
            contemporary_neighbours.append({node_B: time_depth})
  return contemporary_neighbours

def make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary):
  if 'contemporary_neighbour_dictionary.json' in dir:
    return json.load(open('contemporary_neighbour_dictionary.json', 'r'))
  contemporary_neighbour_dictionary = {}
  for i in range(len(trees)):
    tree = trees[i]
    for node in tree.keys():
      contemporary_neighbours = find_contemporary_neighbours(node, trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary)
      contemporary_neighbour_dictionary[node] = contemporary_neighbours  
  json.dump(contemporary_neighbour_dictionary, open('contemporary_neighbour_dictionary.json', 'w'), indent=4)
  return contemporary_neighbour_dictionary

def make_potential_donors(reconstructed_locations_dictionary, time_depths_dictionary, contemporary_neighbour_dictionary):
  if 'potential_donors.json' in dir:
    return json.load(open('potential_donors.json', 'r')) 
  potential_donors = {'node_names': [], 'time_depths': []}
  node_names = []
  branch_lengths = []
  time_depths = []
  number_of_neighbours = []
  for item in reconstructed_locations_dictionary.items():
    if not item[1] == None:
      node_names.append(item[0])
      branch_length = findBranchLength(item[0]) 
      branch_lengths.append(float(branch_length))
      time_depths.append(time_depths_dictionary[item[0]])
      number_of_neighbours.append(float(len(contemporary_neighbour_dictionary[item[0]])))
  together = zip(node_names, branch_lengths, time_depths, number_of_neighbours)
  together_sorted = sorted(together, key = lambda x:x[2])
  potential_donors['node_names'] = [x[0] for x in together_sorted]
  potential_donors['branch_lengths'] = [x[1] for x in together_sorted]
  potential_donors['time_depths'] = [x[2] for x in together_sorted]
  potential_donors['number_of_neighbours'] = [x[3] for x in together_sorted]
  json.dump(potential_donors, open('potential_donors.json', 'w'), indent=4)
  return potential_donors

def make_contact_events(potential_donors, contemporary_neighbour_dictionary, time_depths_dictionary, rate_per_branch_length_per_pair):  
  contact_events = []
  donees = {}
  donors = {}
  together = zip(potential_donors['node_names'], potential_donors['branch_lengths'], potential_donors['time_depths'], potential_donors['number_of_neighbours'])
  for item in together:
    donor = item[0]
    branch_length = item[1]
    number_of_neighbours = int(item[3])
    donor_time_depth = item[2] 
    beginning_time_depth = donor_time_depth
    end_time_depth = beginning_time_depth + branch_length
    event_time = np.random.random()
    event_time = branch_length * event_time
    event_time = event_time + beginning_time_depth   
    probability = rate_per_branch_length_per_pair * branch_length
    sample = np.random.random(number_of_neighbours)
    sample = np.where(sample < probability)[0]
    if len(sample) > 0:    
      donees_to_add = np.array(contemporary_neighbour_dictionary[donor])[sample]
      for donee in donees_to_add:
        contact_events.append({'donor': {donor: donor_time_depth}, 'donee': donee, 'event_time': event_time})
        donees.update(donee)
  contact_events = sorted(contact_events, key = lambda x: x['event_time'])
  donees = {}
  donors = {}
  for contact_event in contact_events:
    donors.update(contact_event['donor'])
    donee = list(contact_event['donee'].keys())[0]
    if not donee in donors:
      donees.update(contact_event['donee'])
  return contact_events, donees

def make_node_value(parent_value, branch_length, substitution_matrix, states):
  if branch_length == 1:
    matrix = np.array(substitution_matrix)
  else:
    matrix = fractional_matrix_power(np.array(substitution_matrix), branch_length)
  parent_value_index = states.index(parent_value)
  row = matrix[parent_value_index]
  node_value_index = np.random.choice(np.arange(0, len(states)), p = row)
  return states[node_value_index]

def choose_root_value(base_frequencies):
  return np.random.choice(list(base_frequencies.keys()), p = list(base_frequencies.values()))

def make_child_dictionary(trees):
  if 'child_dictionary.json' in dir:
    child_dictionary = json.load(open('child_dictionary.json', 'r'))
    return child_dictionary
  child_dictionary = {}
  for i in range(len(trees)):
    tree = trees[i]
    for node in tree:
      children = findChildren(node)
      child_dictionary[node] = children
  json.dump(child_dictionary, open('child_dictionary.json', 'w'), indent=4)
  return child_dictionary
  
def find_children(node, child_dictionary):
  try:
    return child_dictionary[node]
  except:
    print(node)
    print('this should not happen')
    return None


def assign_feature(tree, node, parent_value, substitution_matrix, states, base_frequencies, child_dictionary, to_exclude=[], given_value=None):
  children = find_children(node, child_dictionary)
  if given_value == None:
    if parent_value == None:
      node_value = choose_root_value(base_frequencies)
      tree[node] = node_value
    else:
      branch_length = findBranchLength(node)
      node_value = make_node_value(parent_value, branch_length, substitution_matrix, states)
      tree[node] = node_value
  else:
    tree[node] = given_value
    node_value = given_value
  for child in children:
    if not child in to_exclude:
      tree = assign_feature(tree, child, node_value, substitution_matrix, states, base_frequencies, child_dictionary)
  return tree

def contact_simulation(trees, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, locations, nodes_to_tree_dictionary, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary, contemporary_neighbour_dictionary, potential_donors, child_dictionary):
  contact_events, donees = make_contact_events(potential_donors, contemporary_neighbour_dictionary, time_depths_dictionary, rate_per_branch_length_per_pair)
  json.dump(contact_events, open('contact_events.json', 'w'), indent=4)
  json.dump(donees, open('donees.json', 'w'), indent=4)
  number_of_features = len(substitution_matrix_list)
  final_result_trees = []
  for i in range(len(trees)):
    final_result_trees.append({})
  for j in range(number_of_features):
    substitution_matrix = substitution_matrix_list[j]
    states = states_list[j]
    base_frequencies = base_frequencies_list[j]
    borrowability = borrowability_list[j]
    if borrowability == 1:
      contact_events_sample = deepcopy(contact_events)
      donees_sample = deepcopy(donees)
    else:
      contact_event_sample_probabilities = np.random.random(len(contact_events))
      contact_events_sample_indices = np.where(contact_event_sample_probabilities < borrowability)
      contact_events_sample = deepcopy(contact_events)
      contact_events_sample = np.take(contact_events_sample, contact_events_sample_indices)[0]
      
      donees_sample = {}
      donors_sample = {}
#       print(len(contact_events_sample))
#       print(len(contact_events))
#       print('****')
      for contact_event in contact_events_sample:
#         print(contact_event)
        donors_sample.update(contact_event['donor'])
        donee = list(contact_event['donee'].keys())[0]
        if not donee in donors_sample:
          donees_sample.update(contact_event['donee'])
#     print('****')
#     print(donees == donees_sample)
    for i in range(len(trees)):
      tree = trees[i]
      root = findRoot(tree)
      trees[i] = assign_feature(tree, root, parent_value=None, substitution_matrix=substitution_matrix, states=states, base_frequencies=base_frequencies, child_dictionary=child_dictionary, to_exclude=donees_sample, given_value=None)  
    for contact_event in contact_events_sample:
      donor = list(contact_event['donor'].keys())[0]
      donee = list(contact_event['donee'].keys())[0]
      donor_tree_index = nodes_to_tree_dictionary[donor]
      donee_tree_index = nodes_to_tree_dictionary[donee]
      donor_value = trees[donor_tree_index][donor]
      trees[donee_tree_index] = assign_feature(trees[donee_tree_index], donee, parent_value=None, substitution_matrix=substitution_matrix, states=states, base_frequencies=base_frequencies, child_dictionary=child_dictionary, to_exclude=donees_sample, given_value=donor_value)
      if donee in donees_sample:
        del donees_sample[donee]  
    for i in range(len(trees)):
      tree = trees[i]
      for node in tree:
        value = tree[node]
        if j == 0:
          final_result_trees[i][node] = []
        final_result_trees[i][node].append(value)  
  return final_result_trees

def make_value_dictionary(final_result_trees, list_of_languages):
  value_dictionary = {}
  for i in range(len(final_result_trees)):
    tree = final_result_trees[i]
    for key in tree:
      glottocode = find_glottocode(findNodeNameWithoutStructure(key))
      if glottocode in list_of_languages:
        values = tree[key]
        value_dictionary[glottocode] = [float(x) for x in values]
  return value_dictionary

def make_input_array(value_dictionary):
  result = []
  sorted_keys = sorted(value_dictionary.keys())
  for key in sorted_keys:
    values = value_dictionary[key]
    values = [0 if x == None else x for x in values]
    result.append(values)
  return np.array([result])

def make_output_array(value_dictionary, sample):
  result = []
  for item in sample:
    values = value_dictionary[item]
    values = [0 if x == None else x for x in values]
    result.append(values)
  result = np.array(result)
  shape = np.shape(result)
  result = np.reshape(result, (shape[0], 1, shape[1]))
  return result


def make_input_and_output_arrays(trees, list_of_languages, sample, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, number_of_simulations):
  locations = get_locations(trees)
  nodes_to_tree_dictionary = make_nodes_to_tree_dictionary(trees)
  reconstructed_locations_dictionary = make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary)
  time_depths_dictionary = make_time_depths_dictionary(trees)
  parent_dictionary = make_parent_dictionary(trees)
  contemporary_neighbour_dictionary = make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary)
  potential_donors = make_potential_donors(reconstructed_locations_dictionary, time_depths_dictionary, contemporary_neighbour_dictionary)
  child_dictionary = make_child_dictionary(trees)
  input_array = []
  output_array = []
  for i in range(number_of_simulations):
    trees = contact_simulation(trees, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, locations, nodes_to_tree_dictionary, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary, contemporary_neighbour_dictionary, potential_donors, child_dictionary)
    value_dictionary = make_value_dictionary(trees, list_of_languages)
    to_append_to_input_array = make_input_array(value_dictionary)
    to_append_to_output_array = make_output_array(value_dictionary, sample)
    input_array.append(to_append_to_input_array)
    output_array.append(to_append_to_output_array)
  return input_array, output_array

def make_relatedness_pairs_dictionary(list_of_languages, trees, parent_dictionary):
  f = 'relatedness_pairs_dictionary.json'
  if f in dir:
    return json.load(open(f, 'r'))
  relatedness_pairs_dictionary = {}
  for item_1 in list_of_languages:
    relatedness_pairs_dictionary[item_1] = {}
    for item_2 in list_of_languages:
      relatedness_pairs_dictionary[item_1][item_2] = 'unrelated'
  for i in range(len(trees)):
    tree = trees[i]
    print(i)
    for node_1 in tree:
      glottocode_1 = find_glottocode(findNodeNameWithoutStructure(node_1))
      if glottocode_1 in list_of_languages:
        for node_2 in tree:
          glottocode_2 = find_glottocode(findNodeNameWithoutStructure(node_2)) 
          if glottocode_2 in list_of_languages:     
            phyletic_distance = find_phyletic_distance(node_1, node_2, parent_dictionary)
            relatedness_pairs_dictionary[glottocode_1][glottocode_2] = phyletic_distance
  json.dump(relatedness_pairs_dictionary, open(f, 'w'), indent=4)
  return relatedness_pairs_dictionary

def find_distance(language_1, language_2, df1, df2):
  long_1 = df1['Longitude'][language_1]
  lat_1 = df1['Latitude'][language_1]
  long_2 = df1['Longitude'][language_2]
  lat_2 = df1['Latitude'][language_2]
  if np.isnan(long_1):
    long_1 = df2['longitude'][language_1]
    lat_1 = df2['latitude'][language_1]
  if np.isnan(long_2):
    long_2 = df2['longitude'][language_2]
    lat_2 = df2['latitude'][language_2]
  return haversine(long_1, lat_1, long_2, lat_2)

def make_distance_pairs_dictionary(list_of_languages):
  f = 'distance_pairs_dictionary.json'
  if f in dir:
    return json.load(open(f, 'r'))
  df1 = pd.read_csv('languages.txt', index_col='ID')
  df2 = pd.read_csv('languages_and_dialects_geo.csv', index_col='glottocode') 
  distance_pairs_dictionary = {}
  for language_1 in list_of_languages:
    print(language_1)
    distance_pairs_dictionary[language_1] = {}
    for language_2 in list_of_languages:
      try:
        distance_pairs_dictionary[language_1][language_2] = find_distance(language_1, language_2, df1, df2)
      except:
        distance_pairs_dictionary[language_1][language_2] = 'unknown'
  json.dump(distance_pairs_dictionary, open(f, 'w'), indent=4)
  return distance_pairs_dictionary

def make_relatedness_array(list_of_languages, sample, relatedness_pairs_dictionary):
  result = []
  for item1 in sample:
    temp = []
    for item2 in list_of_languages:
      relatedness = relatedness_pairs_dictionary[item1][item2]
      if relatedness == 'unrelated': 
        temp.append(np.nan)
      else:
        temp.append(relatedness)
    result.append(temp)
  result = [result]
  result = np.array(result)
  return result

def make_distance_array(list_of_languages, sample, distance_pairs_dictionary):
  result = []
  for item1 in sample:
    temp = []
    for item2 in list_of_languages:
      distance = distance_pairs_dictionary[item1][item2]
      if distance == 'unknown':
        distance = np.nan
      temp.append(distance)
    result.append(temp)
  result = [result]
  result = np.array(result)
  return result

def make_one_hot(input, number_of_bins):
  input_shape = np.shape(input)
  a = np.ndarray.flatten(input)
  b = np.zeros((a.size, number_of_bins))
  b[np.arange(a.size),a] = 1
  output_shape = input_shape + (number_of_bins,)
  return np.reshape(b, output_shape)

def preprocess_relatedness_array(relatedness_array, number_of_relatedness_bins):
  maximum = np.nanmax(relatedness_array) 
  bins = np.linspace(0, maximum+1, number_of_relatedness_bins)
  x = np.digitize(relatedness_array, bins)
  x = x - 1
  x = make_one_hot(x, number_of_relatedness_bins)
  return x

def preprocess_distance_array(distance_array, number_of_distance_bins):
  maximum = np.nanmax(distance_array)
  bins = np.linspace(0, maximum+1, number_of_distance_bins)
  x = np.digitize(distance_array, bins)
  x = x - 1
  x = make_one_hot(x, number_of_distance_bins)
  return x
    
def make_all_arrays(trees, list_of_languages, sample, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, number_of_simulations, number_of_relatedness_bins=10, number_of_distance_bins=10):  
  input_array, output_array = make_input_and_output_arrays(trees, list_of_languages, sample, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, number_of_simulations)
  parent_dictionary = make_parent_dictionary(trees)
  relatedness_pairs_dictionary = make_relatedness_pairs_dictionary(list_of_languages, trees, parent_dictionary)
  distance_pairs_dictionary = make_distance_pairs_dictionary(list_of_languages)
  relatedness_array = make_relatedness_array(list_of_languages, sample, relatedness_pairs_dictionary)
  distance_array = make_distance_array(list_of_languages, sample, distance_pairs_dictionary)
  relatedness_array = preprocess_relatedness_array(relatedness_array, number_of_relatedness_bins)
  distance_array = preprocess_distance_array(distance_array, number_of_distance_bins)
  return input_array, output_array, relatedness_array, distance_array







  


  
  
  
  