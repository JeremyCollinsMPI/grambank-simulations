from LikelihoodFunction import *
from TreeFunctions import *
import numpy as np
from general import *
from copy import deepcopy
import multiprocessing
import os
import json

dir = os.listdir('.')

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
  locations = {}
  for tree in trees:
    for key in tree.keys():
      print(key)
      glottocode = find_glottocode(key)
      nodename = prepare_node_name(key)
      if glottocode in df.index:
        lat = df['latitude'][glottocode]
        long = df['longitude'][glottocode]
        if not lat == None and not long == None:
          locations[nodename] = {'lat': lat, 'long': long}
  json.dump(locations, open('locations.json', 'w'), indent=4)
  return locations

def process_longs_part_1(longs):
  for i in range(len(longs)):
    long = longs[i]
    if long < -30:
      long = 360 - long
      longs[i] = long
  return longs

def process_longs_part_2(long):
  if long > 180:
    long = (180 - (long - 180)) * -1
  return long
    
# def make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary):
#   if 'reconstructed_locations_dictionary.json' in dir:
#     return json.load(open('reconstructed_locations_dictionary.json', 'r'))  
#   reconstructed_locations_dictionary = {}
#   nodes_to_check = []
#   for tree in trees:
#     for key in tree.keys():
#       nodename = prepare_node_name(key) 
#       reconstructed_locations_dictionary[nodename] = None
#       try:
#         location = locations[nodename]
#         reconstructed_locations_dictionary[nodename] = location
#         nodes_to_check.append(nodename)
#       except:
#         pass
#   done = False
#   while not done:   
#     node = nodes_to_check[0]
#     print(node)
#     nodename = node
#     tree_index = nodes_to_tree_dictionary[nodename]
#     tree = trees[tree_index]
#     try:
#       parent = findParent(tree, nodename)
#       children = findChildren(parent)
#       siblings = [x for x in children if not x == node]
#       children_locations = []
#       for child in children:
#         location = reconstructed_locations_dictionary[prepare_node_name(child)]
#         if not location == None:
#           children_locations.append(location)
#       lats = [child_location['lat'] for child_location in children_locations]
#       longs = [child_location['long'] for child_location in children_locations]
#       longs = process_longs_part_1(longs)
#       average_lat = np.mean(lats)
#       average_long = np.mean(longs)
#       average_long = process_longs_part_2(average_long)
# #       for sibling in siblings:
# #         reconstructed_locations_dictionary[sibling] = locations[sibling]
#       reconstructed_locations_dictionary[parent] = {'lat': average_lat, 'long': average_long}
#       nodes_to_check.remove(nodes_to_check[0])
# #       for sibling in siblings:
# #         nodes_to_check.remove(sibling)
#       nodes_to_check.append(parent)
#     except:
#       nodes_to_check.remove(node)
#     if len(nodes_to_check) == 0:
#       done = True      
#   json.dump(reconstructed_locations_dictionary, open('reconstructed_locations_dictionary.json', 'w'), indent=4)
#   return reconstructed_locations_dictionary


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
    print(i)
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

def correct_time_depth(A, B, tree, time_depths_dictionary, A_time_depth, parent_dictionary):
  B_time_depth = time_depths_dictionary[B]
  A_parent = find_parent(A, parent_dictionary)
  if A_parent == None:
    A_parent_time_depth = 0
  else:
    A_parent_time_depth = time_depths_dictionary[A_parent]
  if B_time_depth < A_parent_time_depth:
    return False
  if B_time_depth >= A_parent_time_depth and B_time_depth <= A_time_depth:
    return True
  if is_eldest_node_younger_than_A(B, A, tree, time_depths_dictionary, A_time_depth, B_time_depth, parent_dictionary):
    return True
  return False

def find_contemporary_neighbours(node_A, trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary):
  contemporary_neighbours = []
  for i in range(len(trees)):
    tree = trees[i]
    for node_B in tree.keys():
      if not node_A == node_B:
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
  potential_donors = {'node_names': [], 'probabilities': [], 'time_depths': []}
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
  matrix = fractional_matrix_power(np.array(substitution_matrix), branch_length)
  parent_value_index = states.index(parent_value)
  row = matrix[parent_value_index]
  node_value_index = np.random.choice(np.arange(0, len(states)), p = row)
  return states[node_value_index]

def choose_root_value(base_frequencies):
  return np.random.choice(list(base_frequencies.keys()), p = list(base_frequencies.values()))

def assign_feature(tree, node, parent_value, substitution_matrix, states, base_frequencies, to_exclude=[], given_value=None):
  children = findChildren(node)
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
      tree = assign_feature(tree, child, node_value, substitution_matrix, states, base_frequencies)
  return tree

def contact_simulation(trees, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, locations, nodes_to_tree_dictionary, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary, contemporary_neighbour_dictionary, potential_donors):
  contact_events, donees = make_contact_events(potential_donors, contemporary_neighbour_dictionary, time_depths_dictionary, rate_per_branch_length_per_pair)
  for i in range(len(trees)):
    tree = trees[i]
    root = findRoot(tree)
    trees[i] = assign_feature(tree, root, parent_value=None, substitution_matrix=substitution_matrix, states=states, base_frequencies=base_frequencies, to_exclude=donees, given_value=None)
  for contact_event in contact_events:
    donor = list(contact_event['donor'].keys())[0]
    donee = list(contact_event['donee'].keys())[0]
    donor_tree_index = nodes_to_tree_dictionary[donor]
    donee_tree_index = nodes_to_tree_dictionary[donee]
    donor_value = trees[donor_tree_index][donor]
    trees[donee_tree_index] = assign_feature(trees[donee_tree_index], donee, parent_value=None, substitution_matrix=substitution_matrix, states=states, base_frequencies=base_frequencies, to_exclude=donees, given_value=donor_value)
    if donee in donees:
      del donees[donee]  
  return trees

def make_value_dictionary(trees, list_of_languages):
  value_dictionary = {}
  for i in range(len(trees)):
    tree = trees[i]
    for key in tree:      
      glottocode = find_glottocode(findNodeNameWithoutStructure(key))
      print(glottocode)
      if glottocode in list_of_languages:
        value_dictionary[glottocode] = float(tree[key])
  return value_dictionary

def make_input_array(value_dictionary):
  result = []
  sorted_keys = sorted(value_dictionary.keys())
  for key in sorted_keys:
    result.append(value_dictionary[key])
  return result

def make_output_array(value_dictionary, sample):
  result = []
  for item in sample:
    result.append(value_dictionary[item])
  return result

def make_input_and_output_arrays(trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations):
  locations = get_locations(trees)
  nodes_to_tree_dictionary = make_nodes_to_tree_dictionary(trees)
  reconstructed_locations_dictionary = make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary)
  time_depths_dictionary = make_time_depths_dictionary(trees)
  parent_dictionary = make_parent_dictionary(trees)
  contemporary_neighbour_dictionary = make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary)
  potential_donors = make_potential_donors(reconstructed_locations_dictionary, time_depths_dictionary, contemporary_neighbour_dictionary)
  input_array = []
  output_array = []
  for i in range(number_of_simulations):
    trees = contact_simulation(trees, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, locations, nodes_to_tree_dictionary, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary, contemporary_neighbour_dictionary, potential_donors)
    value_dictionary = make_value_dictionary(trees, list_of_languages)
    to_append_to_input_array = make_input_array(value_dictionary)
    to_append_to_output_array = make_output_array(value_dictionary, sample)
    input_array.append(to_append_to_input_array)
    output_array.append(to_append_to_output_array)
  return input_array, output_array




  '''
what else is needed?

you need the relatedness arrays

relatedness_pairs_dictionary

distance_pairs_dictionary

you are then making an array of shape (samples, languages)
then another function for preprocessing it for the model,
which turns it into (1, samples, languages, number of bins)

preprocess_relatedness_array(relatedness_array, number_of_relatedness_bins)

preprocess_distance_array(distance_array, number_of_distance_bins)





def find_relatedness(language_1, language_2, trees):
   .....
   
def find_distance(language_1, language_2, ...


def make_relatedness_pairs_dictionary(list_of_languages, trees):
  for language_1 in list_of_languages:
    for language_2 in list_of_languages:
      relatedness = find_relatedness(language_1, language_2, trees)

def make_distance_pairs_dictionary(list_of_languages, ...)
...

def make_relatedness_array(list_of_languages, sample, relatedness_pairs_dictionary):
  result = []
  for item1 in sample:
    temp = []
    for item2 in list_of_languages:
      relatedness = relatedness_pairs_dictionary[item1][item2]
      temp.append(relatedness)
    result.append(temp)
  return result

def make_distance_array(list_of_languages, sample, distance_pairs_dictionary):
  result = []
  for item1 in sample:
    temp = []
    for item2 in list_of_languages:
      distance = distance_pairs_dictionary[item1][item2]
      temp.append(distance)
    result.append(temp)
  return result

def preprocess_relatedness_array(relatedness_array, number_of_bins):


def preprocess_distance_array(distance_array, number_of_bins):

  
then you need a main function to prepare all of the arrays

e.g.

def make_all_arrays(trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations)
make input and output arrays
make relatedness and distance arrays

then model loads all of those in the training function





  '''



























'''
contact simulation:

you have the trees
you produce the donees
for each tree, you assign feature to the root, excluding the donees
then you go through each contact event; you then assign the given feature to 
the donee in the donee's tree, excluding the other donees
you remove that donee (it's always donees[0]).  continue until there are no donees
left.
some details that are unresolved:
how do you deal with one donee for multiple contact events?  you would assign the feature
to the donee, but that would not be the end of it.  you have to wait
until you are sure that it is not the donee of another contact event.
i think another way of doing it though is that you could just re-assign the feature for that
node, if that happens.  no harm with assigning the features twice.
one complication is that the donee node will be in the list to_exclude in the function assign_feature.
but since that only affects the children of that node, that should not be a problem.


so now need to sketch the functions

trees, contact events and donees as input, as well as whatever dictionaries you need.

for i in range(len(trees)):
  tree = trees[i]
  root = findRoot(tree)
  trees[i] = assign_feature(...tree, ...)

for contact_event in contact_events:
  donor = contact_event['donor'].keys()[0]
  donee = contact_events['donee'].keys()[0]
  donor_tree_index = nodes_to_tree_dictionary[donor]
  donee_tree_index = nodes_to_tree_dictionary[donee]
  donor_value = trees[donor_tree_index][donor]
  trees[donee_tree_index][donee] = assign_feature(....to_exclude=donees, given_value = donor_value)
  donees.pop(0)
  
then what?
you have the trees;
you now need to make into an array.  this could be done by a separate function.
that could be produce_simulated_feature_array.
so you could also still use get_values_for_tree, run over the trees.
except now, get_values_for_tree will only get the values.
there will be a separate function for contact_simulation, with whatever arguments that needs.
trees = contact_simulation(trees, ...)
within that function, you load the dictionaries.
  
  
the idea of donees is to have a list of ones to exclude, to save time.
but you should only exclude a donee if it is not a donor earlier than it is a donee






'''










# def contact_simulation(trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations=1):
#   locations = get_locations(trees)
#   nodes_to_tree_dictionary = make_nodes_to_tree_dictionary(trees)
#   reconstructed_locations_dictionary = make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary)
#   time_depths_dictionary = make_time_depths_dictionary(trees)
#   parent_dictionary = make_parent_dictionary(trees)
#   contemporary_neighbour_dictionary = make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary)
#   potential_donors = make_potential_donors(reconstructed_locations_dictionary, time_depths_dictionary, contemporary_neighbour_dictionary)
#   '''
#   first want to make sure that the trees have some UNASSIGNED or None values for each node.
#   or just use a copy of the trees. but it might be slow.
#   '''
#   input_array = []
#   output_array = []
#   for i in range(number_of_simulations):
#     array = []
#     contact_events, donees = make_contact_events(potential_donors, contemporary_neighbour_dictionary, time_depths_dictionary, rate_per_branch_length_per_pair)
# #     json.dump(contact_events, open('contact_events.json', 'w'), indent=4)
#     for i in range(len(trees)):
#       tree = trees[i]
#       root = findRoot(tree)
#       trees[i] = assign_feature(tree, root, parent_value=None, substitution_matrix=substitution_matrix, states=states, base_frequencies=base_frequencies, to_exclude=donees, given_value=None)
#     for contact_event in contact_events:
#       donor = list(contact_event['donor'].keys())[0]
#       donee = list(contact_event['donee'].keys())[0]
#       donor_tree_index = nodes_to_tree_dictionary[donor]
#       donee_tree_index = nodes_to_tree_dictionary[donee]
#       donor_value = trees[donor_tree_index][donor]
#       trees[donee_tree_index] = assign_feature(trees[donee_tree_index], donee, parent_value=None, substitution_matrix=substitution_matrix, states=states, base_frequencies=base_frequencies, to_exclude=donees, given_value=donor_value)
#       if donee in donees:
#         del donees[donee]  
#     to_append_to_input_array = make_input_array(trees)
#     to_append_to_output_array = make_output_array(trees)
#     
#     
#     
#     if not list_of_languages == None:
#       for tree in trees:
#         keys = tree.keys()
#         sorted_keys = sorted(keys)
#         result = [float(tree[key]) for key in sorted_keys if find_glottocode(findNodeNameWithoutStructure(key)) in list_of_languages]
#         array = array + result
# #     json.dump(trees, open('trees_results.json', 'w'), indent=4)
#     simulated_feature_array = simulated_feature_array + [array]
#   return simulated_feature_array
#   
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# def contact_simulation_writing_to_file(filename, trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations=1):
#   simulated_feature_array = contact_simulation(trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations)
#   np.save(filename, simulated_feature_array)
#   return simulated_feature_array
# 
# def simulate_data(tree, substitution_matrix, states, base_frequencies):
#   root = findRoot(tree)
#   tree = assign_feature(tree, root, parent_value=None, substitution_matrix=substitution_matrix, states=states, base_frequencies=base_frequencies)
#   return tree
#   
# def get_values_for_tree(tree, substitution_matrix, states, base_frequencies, list_of_languages):
#   tree = simulate_data(tree, substitution_matrix, states, base_frequencies)
#   if not list_of_languages == None:
#     keys = tree.keys()
#     sorted_keys = sorted(keys)
#     result = [float(tree[key]) for key in sorted_keys if find_glottocode(findNodeNameWithoutStructure(key)) in list_of_languages]
#   else:
#     tips = findTips(tree)
#     sorted_tips = sorted(tips)
#     result = [float(tree[tip]) for tip in sorted_tips]
#   return result
# 
# def produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations=1, list_of_languages=None):
#   final_array = []
#   print('Producing simulated data')
#   for i in range(number_of_simulations):
#     print('Iteration ', i)
#     array = []
#     for tree in trees:
#       values = get_values_for_tree(tree, substitution_matrix, states, base_frequencies, list_of_languages)
#       array = array + values
#     final_array.append(array)
#   final_array = np.array(final_array)
#   return final_array
# 
# def produce_simulated_feature_array_and_write_to_file(filename, trees, substitution_matrix, states, base_frequencies, number_of_simulations=1, list_of_languages=None):
#   array = produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations, list_of_languages)
#   np.save(filename, array)
# 
# def produce_simulated_feature_array_threaded(trees, substitution_matrix, states, base_frequencies, number_of_simulations=1, list_of_languages=None, number_of_threads=1):
#   simulations_per_thread = int(number_of_simulations / number_of_threads)
#   threads = {}
#   for i in range(number_of_threads):
#     filename = 'result_' + str(i) + '.npy'
#     threads[str(i)] = multiprocessing.Process(target=produce_simulated_feature_array_and_write_to_file, args=(filename, trees, substitution_matrix, states, base_frequencies, simulations_per_thread, list_of_languages))
#     threads[str(i)].start()
#   for i in range(number_of_threads):  
#     threads[str(i)].join()
#   result = np.load('result_0.npy')
#   for i in range(1, number_of_threads):
#     result = np.concatenate([result, np.load('result_' + str(i) + '.npy')])
#   return result
# 
# 
# 
# 
# def make_input(simulated_feature_array):
#   '''
#   you want an array of shape simulations, size of sample, languages-1
#   
#   it would so far be of shape [simulations, languages]
#   
#   the idea is that for each language in the sample, you have the other languages
#   
#   samples is an array of indices
#   
#   so you are taking slices of the array
#   
#   
#   e.g. sfa[0][0] would be language 0 in simulation 0
#   you want sfa[0][0:1600] but without sample[0]
#   then you want sfa[1][0:1600] without sample[0]
#   
#   you could of course just have all languages.  
#   so make something of shape simulations, size of sample, languages
#   
#   so actually just return something of shape simulations, 1, languages.
#   then tf can broadcast it
#   
#   actually will be of shape [1, 1, languages]
#   
#   
#   
#   '''
# 
#   return np.reshape(simulated_feature_array, [np.shape(simulated_feature_array)[0], 1, np.shape(simulated_feature_array)[1]])
# 
# def make_output(simulated_feature_array, samples):
#   '''making something of shape [simulations, samples, 1]
#   so you need to slice sfa by the indices in samples
#   
#   '''  
#   result = np.take(simulated_feature_array, samples, axis=1)
#   number_of_samples = len(samples)
#   number_of_simulations = np.shape(simulated_feature_array)[0]
#   result = np.reshape(result, [number_of_simulations, number_of_samples, 1])
#   return result
  
  
  


  
  
  
  