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
  '''used to be findNodeName, will now just return node'''
  return node
#   return findNodeName(node)

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
      nodename = prepare_node_name(key)
      if glottocode in df.index:
        lat = df['latitude'][glottocode]
        long = df['longitude'][glottocode]
        if not lat == None and not long == None:
          locations[nodename] = {'lat': lat, 'long': long}
  json.dump(locations, open('locations.json', 'w'), indent=4)
  return locations

def process_longs_part_1(longs):
  '''
  any long below -30 should be transformed.
  
  '''

  for i in range(len(longs)):
    long = longs[i]
    if long < -30:
      long = 360 - long
      longs[i] = long
  return longs

def process_longs_part_2(long):
  '''
  transform the long back
  '''
  if long > 180:
    long = (180 - (long - 180)) * -1
  return long
  
    
def make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary):

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
  for tree in trees:
    for key in tree.keys():
      nodename = prepare_node_name(key) 
      reconstructed_locations_dictionary[nodename] = None
      try:
        location = locations[nodename]
        reconstructed_locations_dictionary[nodename] = location
        nodes_to_check.append(nodename)
        # location should be a dictionary of lat and long
      except:
        pass
  done = False
  print('***')
  print(nodes_to_check)
  while not done:
    node = nodes_to_check[0]
#     print('node', node)
    nodename = node
    tree_index = nodes_to_tree_dictionary[nodename]
    tree = trees[tree_index]
    try:
      parent = findParent(tree, nodename)
      children = findChildren(parent)
#       print('children: ', children)
      siblings = [x for x in children if not x == node]
      children_locations = []

      for child in children:
        try:
          children_locations.append(locations[prepare_node_name(child)])
        except:
          pass
      lats = [child_location['lat'] for child_location in children_locations]
      longs = [child_location['long'] for child_location in children_locations]

      '''
      also need to process the longitudes;
      you need to make anything negative into something + 180.
      
      '''
      
      longs = process_longs_part_1(longs)
      
      
      average_lat = np.mean(lats)
      average_long = np.mean(longs)
      average_long = process_longs_part_2(average_long)
      for sibling in siblings:
        reconstructed_locations_dictionary[sibling] = locations[sibling]
      reconstructed_locations_dictionary[parent] = {'lat': average_lat, 'long': average_long}
      nodes_to_check.remove(nodes_to_check[0])
      for sibling in siblings:
        nodes_to_check.remove(sibling)
#         print('removing sibling ', sibling)
      nodes_to_check.append(parent)
#       print('removing parent ', parent)
    except:
      pass
#     print(node)
    nodes_to_check.remove(node)
    if len(nodes_to_check) == 0:
      done = True      
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
  '''
  for tree in trees, you have to take the tips;
  maybe just assume the roots of the trees are roughly equally old.  
  so they all start at 0.  then the more branch lengths you need to get to a node, the later the date.
  
  
  in this case it is easiest to use the full node description, since it uses branch lengths and is easier to handle
  when finding the children.
  '''
  
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



'''
haversine(lon1, lat1, lon2, lat2)
'''
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
  '''
  list of nodes as keys and time depths as values
  
  what do you do if the node is the root of the tree?
  
  you can just set parent time depth to 0.
  
  
  '''
  contemporary_neighbours = []
  for i in range(len(trees)):
    tree = trees[i]
    for node_B in tree.keys():
      if not node_A == node_B:
        A_time_depth = time_depths_dictionary[node_A]
        if correct_time_depth(node_A, node_B, tree, time_depths_dictionary, A_time_depth, parent_dictionary):
          if neighbouring(node_A, node_B, reconstructed_locations_dictionary):
#           print('yes')
#           print(node_A)
#           print(node_B)
            time_depth = time_depths_dictionary[node_B]
            contemporary_neighbours.append({node_B: time_depth})
  return contemporary_neighbours

def make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary):

  '''
   you randomly select branches from the trees.  you also need to select a random time along that branch
  you then use the parent's value, and you assign it to a neighbouring node that is younger than the time that you have selected
  
  so it would be good if the contemporary nodes dictionary also has the time depth 
  
  the idea is to find the neighbours of each node;
  but you also need to take into account time depth;
  anything younger than your time depth (i.e. date is greater)
  but how much younger can it be?
  needs to be older than your children.
  
  
  need to rethink this.
  it is actually the branch length between the parent and the node that you are considering.
  the contact event then happens from the parent node to some other node, with a probability
  which is affected by the branch length.
  you need to look up contemporary neighbours of the parent node.
  so it can't be older than the current node. younger than the parent but not younger than the child.
  so actually it is not as simple as finding 'contemporary neighbours' of a node;
  you're looking for neighbours of a branch.
  
  so for a node in a tree, contemporary neighbours are defined as:
  the geographically neigbouring nodes of the PARENT of the node;
  if the node's time depth > time depth of the parent and < time depth of the node.
  
  
  no, this is not correct.
  
  you also need to take into account B's child I think.
  
  it is any neighbouring node between the time depth of the parent and the time depth of 
  the node; but also the eldest geographically neighbouring
  nodes which are younger than the node.
  
  you also want to include the time depth of these neighbouring nodes, in case.
  
  
  so the structure should be { node : list}.  the list will be of memebers {nodename: time depth}.
  
  
  '''
  if 'contemporary_neighbour_dictionary.json' in dir:
    return json.load(open('contemporary_neighbour_dictionary.json', 'r'))
  contemporary_neighbour_dictionary = {}
  for i in range(len(trees)):
    print(i)
    tree = trees[i]
    for node in tree.keys():
#       print(node)
      contemporary_neighbours = find_contemporary_neighbours(node, trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary)
      contemporary_neighbour_dictionary[node] = contemporary_neighbours  
#       print(contemporary_neighbours)
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
  '''
  an array of dictionaries. this doesn't need to be stored since it is generated in each simulation
  contact_event['donor'] and contact_event['donee']. 
  for doing the contact simulation, you need to have the list of donees;
  you have copies of trees;
  you assign features to them excluding the donees;
  then you find the first donee, assign it the value from the donor;
  to do that you find the donor's name, look up the tree, then tree[donor name] is the value.  
  find the donee name, look up its tree, then tree[donee name] = value.  
  the contact events need to be sorted at the end;
  sorted by what?  by time depth of the donor.
  
  actually this needs to be redone.
  you may want to sample proportional to branch length but also number of neighbours.
  
  so you have now rate_per_branch_length_per_pair
  
  you need to select the donee with a certain probability too.
  so you find the donor;
  then make a random list of length number of neighbours for that node;
  if any is above the probability for that node, then you have that index using where.
  look up its neighbours and with that index return the donee.
  
  the contact event needs to have a time.  then the contact events need to be sorted by time.
  time is sampled from the beginning time depth of the lineage to the end time depth.
  currently it just uses a single value for 'time depth'.
  the beginning time depth is defined as 0 for a root, and 0 + its branch length.
  
  the beginning time depth is given in time depths dictionary, and the end time depth is beginning time depth + branch length
  
  
  
  '''
  
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
#     print(sample)
    sample = np.where(sample < probability)[0]
#     print(sample)
    if len(sample) > 0:    
      donees_to_add = np.array(contemporary_neighbour_dictionary[donor])[sample]
#       print('***')
#       print(donees_to_add)
#       print(donees)
#       donees = np.concatenate((donees, donees_to_add))
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
    node_value = given_value
  for child in children:
    if not child in to_exclude:
      tree = assign_feature(tree, child, node_value, substitution_matrix, states, base_frequencies)
  return tree


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
def contact_simulation(trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations=1):
  locations = get_locations(trees)
  nodes_to_tree_dictionary = make_nodes_to_tree_dictionary(trees)
  reconstructed_locations_dictionary = make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary)
  time_depths_dictionary = make_time_depths_dictionary(trees)
  parent_dictionary = make_parent_dictionary(trees)
  contemporary_neighbour_dictionary = make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary)
  potential_donors = make_potential_donors(reconstructed_locations_dictionary, time_depths_dictionary, contemporary_neighbour_dictionary)
  '''
  first want to make sure that the trees have some UNASSIGNED or None values for each node.
  or just use a copy of the trees. but it might be slow.
  '''
  simulated_feature_array = []
  for i in range(number_of_simulations):
    array = []
    contact_events, donees = make_contact_events(potential_donors, contemporary_neighbour_dictionary, time_depths_dictionary, rate_per_branch_length_per_pair)
#     json.dump(contact_events, open('contact_events.json', 'w'), indent=4)
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
    if not list_of_languages == None:
      for tree in trees:
        keys = tree.keys()
        sorted_keys = sorted(keys)
        result = [float(tree[key]) for key in sorted_keys if find_glottocode(findNodeNameWithoutStructure(key)) in list_of_languages]
        array = array + result
#     json.dump(trees, open('trees_results.json', 'w'), indent=4)
    simulated_feature_array = simulated_feature_array + [array]
  return simulated_feature_array
  
def contact_simulation_writing_to_file(filename, trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations=1):
  simulated_feature_array = contact_simulation(trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations)
  np.save(filename, simulated_feature_array)
  return simulated_feature_array

def simulate_data(tree, substitution_matrix, states, base_frequencies):
  root = findRoot(tree)
  tree = assign_feature(tree, root, parent_value=None, substitution_matrix=substitution_matrix, states=states, base_frequencies=base_frequencies)
  return tree
  
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
  
  
  


  
  
  
  