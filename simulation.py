from LikelihoodFunction import *
from TreeFunctions import *
import numpy as np
from general import *
from copy import deepcopy
import multiprocessing
import os
import json

dir = os.listdir('.')

def make_trees():
  trees = open('trees.txt','r').readlines()
  for i in range(len(trees)):
    tree = trees[i]
    tree = tree.strip('\n')
    tree = createTree(tree)
    trees[i] = tree
  return trees

def make_nodes_to_tree_dictionary(trees):
  '''
  dictionary with node names as keys and an index as value; the index can be used to find the tree
  for 
  '''
  
  if 'nodes_to_tree_dictionary.json' in dir:
    return json.load(open('nodes_to_tree_dictionary.json', 'r'))
  nodes_to_tree_dictionary = {}
  for i in range(len(trees)):
    for key in tree[i].keys():
      node_name = findNodeNameWithoutStructure(key)
      nodes_to_tree_dictionary[node_name] == i
  json.dump(nodes_to_tree_dictionary, open('nodes_to_tree_dictionary.json', 'w'), indent=4)
  return nodes_to_tree_dictionary

def get_locations(trees):
  '''
  you could use a dataframe here
  also probably fine to use a dictionary
  could always produce dataframes from dictionaries at some point anyway. 
  
  this uses Languages.csv, I think.  you could just load it as a dataframe,
  although it would be good to have the node name as an index
  
  some terminal nodes may not have a location
  
  difficulty of whether all nodes will have a location
  
  not really a problem, since nodes without locations are not treated as candidates for borrowing events.
  
  remaining question of what the keys of the dictionary are.  glottocode, or nodename without structure?
  
  makes sense to use glottocodes here, since that is what the locations dataframe provides.  
  but in the reconstructed locations tree, it makes sense to use
  
  probably better to use nodename, rather than nodename without structure, for the tree functions to work
  
  structure is;
  
  
  nodename (including strucure) : {'lat': lat, 'long': long}
  '''
  if 'locations.json' in dir:
    return json.load(open('locations.json', 'r'))
  
  df = pd.read_csv('Languages.csv', index_col='id')
  locations = {}
  for tree in trees:
    for key in tree.keys():
      print(key)
      glottocode = find_glottocode(key)
      if glottocode in df.index:
        lat = df['latitude'][glottocode]
        long = df['longitude'][glottocode]
        if not lat == None and not long == None:
          locations[key] = {'lat': lat, 'long': long}
          print(locations[key])
  json.dump(locations, open('locations.json', 'w'), indent=4)
  return locations
  
def make_reconstructed_locations_dictionary(trees, locations):

  '''
  this uses locations at the tips;
  question of which tips it needs
  
  it is a large dictionary of nodes and locations
  
  it is (maybe) constructed upwards, beginning with any nodes that have locations; so these do not have to be terminal nodes
  
  you assign whatever locations you can to nodes; then you begin with the tips;
  you look at whether it has a location, and if it does, find its siblings.  you find the ancestral location,
  ignoring any unassigned locations in the siblings.  that gives a location to the parent node.
  you may construct a list of nodes_to_check.  these are all nodes which have a location.
  this list is initially any node which is known to have a location.  you can construct this list from the locations dictionary.
  when you assign a location to a parent, then the parent node is appended; but also any siblings already considered can be deleted.
  you continue doing this, going up to the root of each tree.
  when you assign locations to the initial nodes, you then add these to the dictionary (or maybe these are already in the dictionary).
  whenever you add a parent node, you similarly add the node name as a
  the output is the dictionary of node names and locations.
  
  what about any nodes which don't have locations?
  not a problem, since the contact simulation just ignores them.  you only have nodes with locations as potential donors or 
  donees.
  
  question of what the keys are.  
  it makes sense to use nodenames without structure.  this will be different from the locations dictionary, which 
  uses glottocodes.
  so to find the locations of the first nodes in the output tree, you find the glottocode of the nodename if there is one
  
  steps:
  
  1. construct the list nodes_to_check.  this is done from locations.
  reconstructed_locations_dictionary = {}
  nodes_to_check = []
  for tree in trees:
    for key in trees.keys():
      glottocode = find glottocode(key) # whatever this function actually is
      nodename = some function (key)
      reconstructed_locations_dictionary[nodename] = None
      try:
        location = locations[glottocode]
        reconstructed_locations_dictionary[nodename] = location
        nodes_to_check.append(nodename)
        # location should be a dictionary of lat and long
      except:
        pass
        
   code:     
   
  reconstructed_locations_dictionary = {}
  nodes_to_check = []
  for tree in trees.keys():
    glottocode = find_glottocode(key)
    nodename = key 
    reconstructed_locations_dictionary[nodename] = None
    try:
      location = locations[glottocode]
      reconstructed_locations_dictionary[nodename] = location
      nodes_to_check.append(nodename)
        # location should be a dictionary of lat and long
    except:
      pass
     
        
        
  2. in each node to check, find its parent and siblings
  
  
  how does nodes to check work?
  you initially have any node which has a location.
  to look at the node, find parent and siblings;
  you remove siblings from nodes to check, since they are done;
  you assign a location to the parent, and you add that to nodes to check.
  if a node doesn't have a parent, then remove it from nodes to check.
  remove the node from nodes to check.
  
  
  
  '''
  if 'reconstructed_locations_dictionary.json' in dir:
    return json.load(open('reconstructed_locations_dictionary.json', 'r'))
  
  reconstructed_locations_dictionary = {}
  nodes_to_check = []
  for tree in trees.keys():
    glottocode = find_glottocode(key)
    nodename = key 
    reconstructed_locations_dictionary[nodename] = None
    try:
      location = locations[glottocode]
      reconstructed_locations_dictionary[nodename] = location
      nodes_to_check.append(nodename)
        # location should be a dictionary of lat and long
    except:
      pass
  done = False
  while not done:
    node = nodes_to_check[0]
    nodename = node
    tree_index = nodes_to_tree_dictionary[nodename]
    tree = trees[tree_index]
    try:
      parent = findParent(tree, nodename)
      children = findChildren(parent)
      siblings = [x for x in children if not x == node]
      children_locations = []
      try:
        for child in children:
          children_locations.append(locations[child])
      except:
        pass
      average_lat = np.mean([child_location['lat'] for child_location in children_locations])
      average_long = np.mean([child_location['long'] for child_location in children_locations])
      for sibling in siblings:
        reconstructed_locations_dictionary[sibling] = locations[sibling]
      reconstructed_locations_dictionary[parent] = {'lat': average_lat, 'long': average_long}
      nodes_to_check.remove(nodes_to_check[0])
      for sibling in siblings:
        nodes_to_check.remove(sibling)
      nodes_to_check.append(parent)
    except:
      pass
    nodes_to_check.remove(node)
    if len(nodes_to_check) == 0:
      done = True      
  json.dump(reconstructed_locations_dictionary, open('reconstructed_locations_dictionary.json', 'w'), indent=4)
  return reconstructed_locations_dictionary

  
# def make_time_depths_dictionary(trees):
#   if 'time_depths_dictionary.json' in dir:
#     return json.load(open('time_depths_dictionary.json', 'r'))
#   
#   json.dump(time_depths_dictionary, open('time_depths_dictionary.json', 'w'), indent=4)
#   return time_depths_dictionary
#   
# def make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary):
#   ...
#   '''
#    you randomly select branches from the trees.  you also need to select a random time along that branch
#   you then use the parent's value, and you assign it to a neighbouring node that is younger than the time that you have selected
#   
#   so it would be good if the contemporary nodes dictionary also has the time depth 
#   '''
#   if 'contemporary_neighbour_dictionary.json' in dir:
#     return json.load(open('contemporary_neighbour_dictionary.json', 'r'))
#   
#   json.dump(contemporary_neighbour_dictionary, open('contemporary_neighbour_dictionary.json', 'w'), indent=4)
#   return contemporary_neighbour_dictionary
#  
# 
# 
# def make_potential_donors(trees):
#   ...
#   '''
#   array of node names and an array of probabilities.
#   could return this as a dictionary, with keys 'node_names' and 'probabilities'
#   '''
#   
#   if 'potential_donors.json' in dir:
#     return json.load(open('potential_donors.json', 'r'))
#   
#   json.dump(potential_donors, open('potential_donors.json', 'w'), indent=4)
#   return potential_donors
# 
#   
# 
# 
# def make_contact_events(potential_donors, contemporary_neighbour_dictionary, ...):
#   '''
#   an array of dictionaries. this doesn't need to be stored since it is generated in each simulation
#   '''
#   contact_events = None
#   return contact_events  
# 
# def make_donees(contact_events):
#   '''
#   probably just an array.  
#   '''
#   donees = None
#   return donees

def make_node_value(parent_value, branch_length, substitution_matrix, states):
  matrix = fractional_matrix_power(np.array(substitution_matrix), branch_length)
  parent_value_index = states.index(parent_value)
  row = matrix[parent_value_index]
  node_value_index = np.random.choice(np.arange(0, len(states)), p = row)
  return states[node_value_index]

def choose_root_value(base_frequencies):
  return np.random.choice(list(base_frequencies.keys()), p = list(base_frequencies.values()))

def assign_feature(tree, node, parent_value, substitution_matrix, states, base_frequencies, to_exclude=[], given_value=None):
  children = findChildren(node)
  '''
  could make this more efficient by checking the descendent tips to see whether they are included in the list of languages
  
  in fact you now need a list to_exclude anyway, for when simulating contact
  
  '''
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
  for child in children:
    if not child in to_exclude:
      tree = assign_feature(tree, child, node_value, substitution_matrix, states, base_frequencies)
  return tree

def simulate_data(tree, substitution_matrix, states, base_frequencies):
  root = findRoot(tree)
  tree = assign_feature(tree, root, parent_value=None, substitution_matrix=substitution_matrix, states=states, base_frequencies=base_frequencies)
  return tree

def make_one_hot(state, states):
  index = states.index(state)
  result = rep(0.0, len(states))
  result[index] = 1.0
  return result
  

def get_values_for_tree(tree, substitution_matrix, states, base_frequencies, list_of_languages):
  tree = simulate_data(tree, substitution_matrix, states, base_frequencies)
  if not list_of_languages == None:
    keys = tree.keys()
    sorted_keys = sorted(keys)
    result = [float(tree[key]) for key in sorted_keys if find_glottocode(findNodeNameWithoutStructure(key)) in list_of_languages]
  else:
    tips = findTips(tree)
    sorted_tips = sorted(tips)
    result = [float(tree[tip]) for tip in sorted_tips]
  return result

def produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations=1, list_of_languages=None):
  final_array = []
  print('Producing simulated data')
  for i in range(number_of_simulations):
    print('Iteration ', i)
    array = []
    for tree in trees:
      values = get_values_for_tree(tree, substitution_matrix, states, base_frequencies, list_of_languages)
      array = array + values
    final_array.append(array)
  final_array = np.array(final_array)
  return final_array

def produce_simulated_feature_array_and_write_to_file(filename, trees, substitution_matrix, states, base_frequencies, number_of_simulations=1, list_of_languages=None):
  array = produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations, list_of_languages)
  np.save(filename, array)

def produce_simulated_feature_array_threaded(trees, substitution_matrix, states, base_frequencies, number_of_simulations=1, list_of_languages=None, number_of_threads=1):
  simulations_per_thread = int(number_of_simulations / number_of_threads)
  threads = {}
  for i in range(number_of_threads):
    filename = 'result_' + str(i) + '.npy'
    threads[str(i)] = multiprocessing.Process(target=produce_simulated_feature_array_and_write_to_file, args=(filename, trees, substitution_matrix, states, base_frequencies, simulations_per_thread, list_of_languages))
    threads[str(i)].start()
  for i in range(number_of_threads):  
    threads[str(i)].join()
  result = np.load('result_0.npy')
  for i in range(1, number_of_threads):
    result = np.concatenate([result, np.load('result_' + str(i) + '.npy')])
  return result




def make_input(simulated_feature_array):
  '''
  you want an array of shape simulations, size of sample, languages-1
  
  it would so far be of shape [simulations, languages]
  
  the idea is that for each language in the sample, you have the other languages
  
  samples is an array of indices
  
  so you are taking slices of the array
  
  
  e.g. sfa[0][0] would be language 0 in simulation 0
  you want sfa[0][0:1600] but without sample[0]
  then you want sfa[1][0:1600] without sample[0]
  
  you could of course just have all languages.  
  so make something of shape simulations, size of sample, languages
  
  so actually just return something of shape simulations, 1, languages.
  then tf can broadcast it
  
  actually will be of shape [1, 1, languages]
  
  
  
  '''

  return np.reshape(simulated_feature_array, [np.shape(simulated_feature_array)[0], 1, np.shape(simulated_feature_array)[1]])

def make_output(simulated_feature_array, samples):
  '''making something of shape [simulations, samples, 1]
  so you need to slice sfa by the indices in samples
  
  '''  
  result = np.take(simulated_feature_array, samples, axis=1)
  number_of_samples = len(samples)
  number_of_simulations = np.shape(simulated_feature_array)[0]
  result = np.reshape(result, [number_of_simulations, number_of_samples, 1])
  return result
  
  
  


  
  
  
  