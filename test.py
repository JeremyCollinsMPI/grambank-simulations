from TreeFunctions import *
from simulation import *
import json
from preprocessing_for_grambank import *
from model import *
import os


np.random.seed(10)

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

def test1():
  '''
  making a tree from a glottolog tree string
  
  '''
  trees = open('trees.txt','r').readlines()
  tree_number = 66
  tree = trees[tree_number]
  tree = tree.strip('\n')
  tree = createTree(tree)
  print(tree)

def test2():
  '''
  begin at the root with a particular feature value;
  find the daughters of the root, the length of the branch connecting them,
  have a substitution matrix, and assign them feature values;
  do the same with the daughters of those daughters, etc. until everything in the tree is assigned.
  '''
  trees = open('trees.txt','r').readlines()
  tree_number = 175
  tree = trees[tree_number]
  tree = tree.strip('\n')
  tree = createTree(tree)
  root = findRoot(tree)
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}
  tree = simulate_data(tree, substitution_matrix, states, base_frequencies)
  json.dump(tree, open('result.json', 'w'), indent=4)

def test3():
  trees = open('trees.txt','r').readlines()
#   list_of_languages = get_languages_in_grambank()
#   trees = only_retain_included_languages(trees, list_of_languages)
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}
  simulated_feature_array = produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations=1)
  print(np.shape(simulated_feature_array))


def test4():
  trees = open('trees.txt','r').readlines()
#   list_of_languages = get_languages_in_grambank()
#   trees = only_retain_included_languages(trees, list_of_languages)
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}
  simulated_feature_array = produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations=500)
  na_array = np.ones_like(simulated_feature_array)
  '''
  need the na array
  make np.ones_like(simulated_feature_array)
  number_of_features = 1
  number_of_states = 1
  '''
  
  autoencoder = Autoencoder(simulated_feature_array, na_array, number_of_encoding_weights=30)  
  autoencoder.train(steps=1500)
  test_data = produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations=10)
  autoencoder.show_loss(test_data)

def test5():
  if not 'simulated_feature_array.npy' in os.listdir('.'):
    trees = open('trees.txt','r').readlines()
    substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
    states = ['0', '1']
    base_frequencies = {'0': 1, '1': 0}
    simulated_feature_array = produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations=200)
    na_array = np.ones_like(simulated_feature_array)
    np.save('simulated_feature_array.npy', simulated_feature_array)
  else:
    simulated_feature_array = np.load('simulated_feature_array.npy')
    na_array = np.ones_like(simulated_feature_array)
  autoencoder = Autoencoder(simulated_feature_array, na_array, number_of_encoding_weights=400)  
  autoencoder.train(steps=1500)
  trees = open('trees.txt','r').readlines()
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}

  test_data = produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations=10)
  autoencoder.show_loss(test_data)

def test6():
  number_of_simulations = 20
  number_of_samples = 200
  list_of_languages = get_languages_in_grambank()  
  if not 'simulated_feature_array.npy' in os.listdir('.'):
    trees = open('trees.txt','r').readlines()
    substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
#     substitution_matrix = [[0.98, 0.02], [0.02, 0.98]]

    states = ['0', '1']
    base_frequencies = {'0': 1, '1': 0}
    simulated_feature_array = produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations=number_of_simulations, list_of_languages=list_of_languages)
    na_array = np.ones_like(simulated_feature_array)
    np.save('simulated_feature_array.npy', simulated_feature_array)
  else:
    simulated_feature_array = np.load('simulated_feature_array.npy')
    na_array = np.ones_like(simulated_feature_array)
  number_of_languages = np.shape(simulated_feature_array)[1]
  sample = np.random.choice(np.arange(0, np.shape(simulated_feature_array)[1]), number_of_samples, replace=False)
  input = make_input(simulated_feature_array)
  output = make_output(simulated_feature_array, sample)
  model = Model(number_of_simulations, number_of_samples, number_of_languages)
  model.train(input, output)
  trees = open('trees.txt','r').readlines()
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  substitution_matrix = [[0.98, 0.02], [0.02, 0.98]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}
  test_data = produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations=10, list_of_languages=list_of_languages)
  test_input = make_input(test_data)
  test_output = make_output(test_data, sample)
  model.show_loss(test_input, test_output)
  model.show_intercept()

def test7():

  number_of_simulations = 20
  number_of_samples = 200
  list_of_languages = get_languages_in_grambank()  
  trees = open('trees.txt','r').readlines()
  for i in range(len(trees)):
    tree = trees[i]
    tree = tree.strip('\n')
    tree = createTree(tree)
    trees[i] = tree
  if not 'simulated_feature_array.npy' in os.listdir('.'):
    
    substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
    states = ['0', '1']
    base_frequencies = {'0': 1, '1': 0}
    simulated_feature_array = produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations=number_of_simulations, list_of_languages=list_of_languages)
    number_of_languages = np.shape(simulated_feature_array)[1]
    na_array_1 = np.ones([1, number_of_samples, 1])
    na_array_2 = np.ones([1, 1, number_of_languages])
    np.save('simulated_feature_array.npy', simulated_feature_array)
  else:
    simulated_feature_array = np.load('simulated_feature_array.npy')
    number_of_languages = np.shape(simulated_feature_array)[1]
    na_array_1 = np.ones([1, number_of_samples, 1])
    na_array_2 = np.ones([1, 1, number_of_languages])
  
  sample = np.random.choice(np.arange(0, np.shape(simulated_feature_array)[1]), number_of_samples, replace=False)
  input = make_input(simulated_feature_array)
  output = make_output(simulated_feature_array, sample)
  model = Model(number_of_simulations, number_of_samples, number_of_languages)
  model.train(input, output, na_array_1, na_array_2)
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  substitution_matrix = [[0.98, 0.02], [0.02, 0.98]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}
  test_data = produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations=10, list_of_languages=list_of_languages)
  test_input = make_input(test_data)
  test_output = make_output(test_data, sample)
  model.show_loss(test_input, test_output, na_array_1, na_array_2)
  model.show_intercept()

@timeit
def test8_simulation(trees):
  number_of_simulations = 24
  number_of_threads = 6
  list_of_languages = get_languages_in_grambank()  
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}
#   simulated_feature_array = produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations=number_of_simulations, list_of_languages=None)
  simulated_feature_array = produce_simulated_feature_array_threaded(trees, substitution_matrix, states, base_frequencies, number_of_simulations=number_of_simulations, list_of_languages=list_of_languages, number_of_threads=number_of_threads)
  return simulated_feature_array

def test8():
  trees = open('trees.txt','r').readlines()
  for i in range(len(trees)):
    tree = trees[i]
    tree = tree.strip('\n')
    tree = createTree(tree)
    trees[i] = tree
  simulated_feature_array = test8_simulation(trees)
  np.save('simulated_feature_array.npy', simulated_feature_array)
  print(np.shape(simulated_feature_array))

def test9():
  '''
  will have to write a new function for reconstructing locations in the trees
    
  when you run the simulation, you have trees;
  you are choosing branches of trees randomly, but also taking their branch length into account;
  then you assign values vertically for any nodes which are not donees.
  you take any donor nodes; you then look in a dictionary for that node and find its location.
  you then need to find nearby nodes.  it would be best if there is another dictionary that you can use,
  where you look up the node in that dictionary and it returns a list of nearby nodes, and you pick one randomly.
  so you want to first make a dictionary with reconstructed locations.  you also want a dictionary with time depth for each node.
  then you want to use those to make a dictionary of nearby contemporary nodes.
  
  you randomly select branches from the trees.  you also need to select a random time along that branch
  you then use the parent's value, and you assign it to a neighbouring node that is younger than the time that you have selected
  
  so it would be good if the contemporary nodes dictionary also has the time depth 
  
  how do you select random branches?
  you select the parent node, since that also has the branch length information.
  
  so it is simply selecting a non-terminal node.
  
  you want to choose some donor nodes; you then find their donees;
  so you're making a dictionary of donors as keys and donees as values. or a list of lists.
  
  you simulate values for any node which is not a donee.  then you have to go through donor nodes in order
  of time depth.  
  
  so you want a list of lists: donor, donee, time depth.  then sorted by time depth.
  
  you could also better make it donor, time depth, donee.  since you are choosing the donor and time depth first, then appending
  donee.
  
  call this list contact_events
  
  for contact_event in contact_events:
    
    you then assign the value of the contact_event[0] to contact_event[2]
    
    you could also use a dictionary here.  contact_event['donor'] and contact_event['donee']
    
    will do it that way.
    
    
    you also need to continually update some descendant nodes after checking the contact_event.
   
    you do this with assign_feature for the donee node, and using given_value = the value that you are assigning.
   
    so you do this:
    
    donated_value = donor_tree[donor]
    
    assign_feature(   ... something like donee, 
    
    donee_tree, donee, None, substitution_matrix, states, base_frequencies, to_exclude=donees, given_value = donated_values
    
    assign_feature(tree, node, parent_value, substitution_matrix, states, base_frequencies, to_exclude=[], given_value=None
  
    so you also need a quick way of finding donor and donee trees
    
    
    so you also want a way of specifying these in the contact_events list.
    
    
    so each member of contact_events could have the keys:
    
    donor, donee, time_depth, donor_tree, donee_tree
    
    one way of generating this list would be to have a list of potential donors; you pick one randomly, 
    taking branch length into account too.
    
    you need a tree mapping nodes onto the trees which they belong to.
    
    call it nodes_to_tree_dictionary.
    so in fact you don't need to have donor_tree and donee_tree as keys in each contact_event.
    you find donor_tree with nodes_to_tree_dictionary[donor] and donee_tree = nodes_to_tree_dictionary[donee]
    
    so you have to find the tree in trees which is the donee tree; 
    
    you can do something like
    
    donee_tree = nodes_to_tree_dictionary[donee]
    donee_tree = assign_feature(donee_tree, node, None, substitution_matrix, states, base_frequencies, to_exclude=donees, given_value=donated_value)
    
    you also want to make a list called donees.  this can be done with make_donees(contact_events).
  
  '''



  trees = open('trees.txt').readlines()
  trees = make_trees()
  locations = get_locations(trees)
  print(locations)
#   reconstructed_locations_dictionary = make_reconstructed_locations_dictionary(trees, locations)
#   time_depths_dictionary = make_time_depths_dictionary(trees)
#   contemporary_neighbours_dictionary = make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary)
#   
#   
#   
#   
# 
# 
# 
#   dataFrame = assignIsoValues('language.csv')
#   result = reconstructLocationsForAllTrees(trees, dataFrame, numberUpTo = 'all', limitToIsos = True)
#   outputFile = open('reconstructedLocations.txt','w')
#   for member in result:
#     outputFile.write(member + '\t' + str(result[member]) + '\n')





test9()






# def test8():
#   '''
#   making a pipeline for submitting a feature
#   
#   it is a function which takes a feature array (simulated or the real data), and returns the loss, given a setting of stability
#   
#   
#   
#   '''
#   stability_value = 0.98
#   test_data = produce_simulated_feature_array(trees, substitution_matrix, states, base_frequencies, number_of_simulations=10, list_of_languages=list_of_languages)
#   test_input = make_input(test_data)
#   test_output = make_output(test_data, sample)
#   show_loss_for_stability_value(feature_array, stability_value)
  







