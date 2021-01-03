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
  
  so it is simply selecting a non-terminal node [actually it could also be a terminal node?]
  
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
  nodes_to_tree_dictionary = make_nodes_to_tree_dictionary(trees)
  reconstructed_locations_dictionary = make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary)
  time_depths_dictionary = make_time_depths_dictionary(trees)
  parent_dictionary = make_parent_dictionary(trees)
  contemporary_neighbour_dictionary = make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary)
  potential_donors = make_potential_donors(reconstructed_locations_dictionary, time_depths_dictionary, contemporary_neighbour_dictionary)
  contact_events, donees = make_contact_events(potential_donors, contemporary_neighbour_dictionary, time_depths_dictionary, rate_per_branch_length_per_pair=0.01)
  print(contact_events)
  print('*******')
  print(donees)
  
def test10():
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}
  simulated_feature_array = contact_simulation(trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=0.01, number_of_simulations=10) 
  print(np.shape(simulated_feature_array))

def test11():
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  number_of_samples = 200
  number_of_simulations = 10
  rate_per_branch_length_per_pair = 0.01 
  if not 'simulated_feature_array.npy' in os.listdir('.'):
    substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
    states = ['0', '1']
    base_frequencies = {'0': 1, '1': 0}
    simulated_feature_array = contact_simulation_writing_to_file('simulated_feature_array.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate_per_branch_length_per_pair, number_of_simulations=number_of_simulations) 
    number_of_languages = np.shape(simulated_feature_array)[1]
    na_array_1 = np.ones([1, number_of_samples, 1])
    na_array_2 = np.ones([1, 1, number_of_languages])
  else:
    simulated_feature_array = np.load('simulated_feature_array.npy')
    number_of_languages = np.shape(simulated_feature_array)[1]
    na_array_1 = np.ones([1, number_of_samples, 1])
    na_array_2 = np.ones([1, 1, number_of_languages])
  print(np.shape(simulated_feature_array))
  sample = np.random.choice(np.arange(0, np.shape(simulated_feature_array)[1]), number_of_samples, replace=False)
  input = make_input(simulated_feature_array)
  output = make_output(simulated_feature_array, sample)
  model = Model(number_of_simulations, number_of_samples, number_of_languages)
  model.train(input, output, na_array_1, na_array_2)

def test12():
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  number_of_samples = 200
  number_of_simulations = 30
  rate_per_branch_length_per_pair = 0.01 
  if not 'simulated_feature_array_training_001.npy' in os.listdir('.'):
    substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
    states = ['0', '1']
    base_frequencies = {'0': 1, '1': 0}
    simulated_feature_array = contact_simulation_writing_to_file('simulated_feature_array_training_001.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate_per_branch_length_per_pair, number_of_simulations=number_of_simulations) 
    number_of_languages = np.shape(simulated_feature_array)[1]
    na_array_1 = np.ones([1, number_of_samples, 1])
    na_array_2 = np.ones([1, 1, number_of_languages])
  else:
    simulated_feature_array = np.load('simulated_feature_array_training_001.npy')
    number_of_languages = np.shape(simulated_feature_array)[1]
    na_array_1 = np.ones([1, number_of_samples, 1])
    na_array_2 = np.ones([1, 1, number_of_languages])
  if not 'simulated_feature_array_test_001.npy' in os.listdir('.'):
    substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
    states = ['0', '1']
    base_frequencies = {'0': 1, '1': 0}
    test_001 = contact_simulation_writing_to_file('simulated_feature_array_test_001.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=0.01, number_of_simulations=number_of_simulations) 
  else:
    test_001 = np.load('simulated_feature_array_test_001.npy')
  if not 'simulated_feature_array_test_002.npy' in os.listdir('.'):
    substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
    states = ['0', '1']
    base_frequencies = {'0': 1, '1': 0}
    test_002 = contact_simulation_writing_to_file('simulated_feature_array_test_002.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=0.02, number_of_simulations=number_of_simulations) 
  else:
    test_002 = np.load('simulated_feature_array_test_002.npy')
  if not 'simulated_feature_array_training_002.npy' in os.listdir('.'):
    substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
    states = ['0', '1']
    base_frequencies = {'0': 1, '1': 0}
    training_002 = contact_simulation_writing_to_file('simulated_feature_array_training_002.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=0.02, number_of_simulations=number_of_simulations) 
  else:
    training_002 = np.load('simulated_feature_array_training_002.npy')
  sample = np.random.choice(np.arange(0, np.shape(simulated_feature_array)[1]), number_of_samples, replace=False)
  input = make_input(simulated_feature_array)
  output = make_output(simulated_feature_array, sample)
  model = Model(number_of_simulations, number_of_samples, number_of_languages)
  model.train(input, output, na_array_1, na_array_2)
  test_input = make_input(test_001)
  test_output = make_output(test_001, sample)
  test_001_001_loss = model.show_loss(test_input, test_output, na_array_1, na_array_2)
  test_input = make_input(test_002)
  test_output = make_output(test_002, sample)
  test_001_002_loss = model.show_loss(test_input, test_output, na_array_1, na_array_2)
  input = make_input(training_002)
  output = make_output(training_002, sample)
  model.train(input, output, na_array_1, na_array_2)
  test_input = make_input(test_001)
  test_output = make_output(test_001, sample)
  test_002_001_loss = model.show_loss(test_input, test_output, na_array_1, na_array_2)
  test_input = make_input(test_002)
  test_output = make_output(test_002, sample)
  test_002_002_loss = model.show_loss(test_input, test_output, na_array_1, na_array_2)
  print('001 001 ', test_001_001_loss)
  print('001 002 ', test_001_002_loss)
  print('002 001 ', test_002_001_loss)
  print('002 002 ', test_002_002_loss)

def test13():
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  number_of_samples = 600
  number_of_simulations = 30
  rate_per_branch_length_per_pair = 0.01 
  rate2 = 0.3
  if not 'simulated_feature_array_training_001.npy' in os.listdir('.'):
    substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
    states = ['0', '1']
    base_frequencies = {'0': 1, '1': 0}
    simulated_feature_array = contact_simulation_writing_to_file('simulated_feature_array_training_001.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate_per_branch_length_per_pair, number_of_simulations=number_of_simulations) 
    number_of_languages = np.shape(simulated_feature_array)[1]
    na_array_1 = np.ones([1, number_of_samples, 1])
    na_array_2 = np.ones([1, 1, number_of_languages])
  else:
    simulated_feature_array = np.load('simulated_feature_array_training_001.npy')
    number_of_languages = np.shape(simulated_feature_array)[1]
    na_array_1 = np.ones([1, number_of_samples, 1])
    na_array_2 = np.ones([1, 1, number_of_languages])
  if not 'simulated_feature_array_test_001.npy' in os.listdir('.'):
    substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
    states = ['0', '1']
    base_frequencies = {'0': 1, '1': 0}
    test_001 = contact_simulation_writing_to_file('simulated_feature_array_test_001.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=0.01, number_of_simulations=number_of_simulations) 
  else:
    test_001 = np.load('simulated_feature_array_test_001.npy')
  if not 'simulated_feature_array_test_010.npy' in os.listdir('.'):
    substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
    states = ['0', '1']
    base_frequencies = {'0': 1, '1': 0}
    test_002 = contact_simulation_writing_to_file('simulated_feature_array_test_010.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate2, number_of_simulations=number_of_simulations) 
  else:
    test_002 = np.load('simulated_feature_array_test_010.npy')
  if not 'simulated_feature_array_training_010.npy' in os.listdir('.'):
    substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
    states = ['0', '1']
    base_frequencies = {'0': 1, '1': 0}
    training_002 = contact_simulation_writing_to_file('simulated_feature_array_training_010.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate2, number_of_simulations=number_of_simulations) 
  else:
    training_002 = np.load('simulated_feature_array_training_010.npy')
  sample = np.random.choice(np.arange(0, np.shape(simulated_feature_array)[1]), number_of_samples, replace=False)
  input = make_input(simulated_feature_array)
  output = make_output(simulated_feature_array, sample)
  model = Model(number_of_simulations, number_of_samples, number_of_languages)
  model.train(input, output, na_array_1, na_array_2)
  test_input = make_input(test_001)
  test_output = make_output(test_001, sample)
  test_001_001_loss = model.show_loss(test_input, test_output, na_array_1, na_array_2)
  test_input = make_input(test_002)
  test_output = make_output(test_002, sample)
  test_001_002_loss = model.show_loss(test_input, test_output, na_array_1, na_array_2)
  input = make_input(training_002)
  output = make_output(training_002, sample)
  model.train(input, output, na_array_1, na_array_2)
  test_input = make_input(test_001)
  test_output = make_output(test_001, sample)
  test_002_001_loss = model.show_loss(test_input, test_output, na_array_1, na_array_2)
  test_input = make_input(test_002)
  test_output = make_output(test_002, sample)
  test_002_002_loss = model.show_loss(test_input, test_output, na_array_1, na_array_2)
  print('001 001 ', test_001_001_loss)
  print('001 010 ', test_001_002_loss)
  print('010 001 ', test_002_001_loss)
  print('010 010 ', test_002_002_loss)

def test14():
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  number_of_samples = 600
  number_of_simulations = 1
  rate1 = 0
  rate2 = 0.3
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}
  contact_simulation_writing_to_file('simulated_feature_array_test_001.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate1, number_of_simulations=number_of_simulations)

def test15():
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  number_of_samples = 900
  number_of_simulations = 60
  test_number_of_simulations = 10
  rate1 = 0.01 
  rate2 = 0.3
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  states = ['0', '1']
  base_frequencies = {'0': 0.5, '1': 0.5}

  if not 'simulated_feature_array_training_001.npy' in os.listdir('.'):
    simulated_feature_array = contact_simulation_writing_to_file('simulated_feature_array_training_001.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate1, number_of_simulations=number_of_simulations) 
    number_of_languages = np.shape(simulated_feature_array)[1]
    na_array_1 = np.ones([1, number_of_samples, 1])
    na_array_2 = np.ones([1, 1, number_of_languages])
  else:
    simulated_feature_array = np.load('simulated_feature_array_training_001.npy')
    number_of_languages = np.shape(simulated_feature_array)[1]
    na_array_1 = np.ones([1, number_of_samples, 1])
    na_array_2 = np.ones([1, 1, number_of_languages])
  if not 'simulated_feature_array_test_001.npy' in os.listdir('.'):
    test_001 = contact_simulation_writing_to_file('simulated_feature_array_test_001.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate1, number_of_simulations=test_number_of_simulations) 
  else:
    test_001 = np.load('simulated_feature_array_test_001.npy')
  if not 'simulated_feature_array_test_002.npy' in os.listdir('.'):
    test_002 = contact_simulation_writing_to_file('simulated_feature_array_test_002.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate2, number_of_simulations=test_number_of_simulations) 
  else:
    test_002 = np.load('simulated_feature_array_test_002.npy')
  if not 'simulated_feature_array_training_002.npy' in os.listdir('.'):
    training_002 = contact_simulation_writing_to_file('simulated_feature_array_training_002.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate2, number_of_simulations=number_of_simulations) 
  else:
    training_002 = np.load('simulated_feature_array_training_002.npy')
  sample = np.random.choice(np.arange(0, np.shape(simulated_feature_array)[1]), number_of_samples, replace=False)
  training_001 = simulated_feature_array
  input = make_input(training_001)
  output = make_output(training_001, sample)
  model = Model(number_of_simulations, number_of_samples, number_of_languages)
  model.train(input, output, na_array_1, na_array_2)
  weights = model.show_weights()
  np.save('weights.npy', weights)
  test_input = make_input(test_001)
  test_output = make_output(test_001, sample)
  test_001_001_loss = model.show_loss(test_input, test_output, na_array_1, na_array_2)
  
  test_input = make_input(test_002)
  test_output = make_output(test_002, sample)
  test_001_002_loss = model.show_loss(test_input, test_output, na_array_1, na_array_2)
  input = make_input(training_002)
  output = make_output(training_002, sample)
  model.train(input, output, na_array_1, na_array_2)
  test_input = make_input(test_001)
  test_output = make_output(test_001, sample)
  test_002_001_loss = model.show_loss(test_input, test_output, na_array_1, na_array_2)
  test_input = make_input(test_002)
  test_output = make_output(test_002, sample)
  test_002_002_loss = model.show_loss(test_input, test_output, na_array_1, na_array_2)
  print('001 001 ', test_001_001_loss)
  print('001 002 ', test_001_002_loss)
  print('002 001 ', test_002_001_loss)
  print('002 002 ', test_002_002_loss)


def test_two_models(borrowing_rate_1, borrowing_rate_2):
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  number_of_samples = 900
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
  number_of_simulations = 1
  rate1 = borrowing_rate_1
  rate2 = borrowing_rate_2
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}
  input_array, output_array = make_input_and_output_arrays(trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate1, number_of_simulations)
  print(np.shape(input_array))
  print(np.shape(output_array))
#   locations = get_locations(trees)
#   nodes_to_tree_dictionary = make_nodes_to_tree_dictionary(trees)
#   reconstructed_locations_dictionary = make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary)
#   time_depths_dictionary = make_time_depths_dictionary(trees)
#   parent_dictionary = make_parent_dictionary(trees)
#   contemporary_neighbour_dictionary = make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary)
#   potential_donors = make_potential_donors(reconstructed_locations_dictionary, time_depths_dictionary, contemporary_neighbour_dictionary)
#   trees = contact_simulation(trees, substitution_matrix, states, base_frequencies, rate1, locations, nodes_to_tree_dictionary, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary, contemporary_neighbour_dictionary, potential_donors)  
#   value_dictionary = make_value_dictionary(trees, list_of_languages)
#   print(value_dictionary)
#   input_array = make_input_array(value_dictionary)
#   print(np.shape(input_array))
#   output_array = make_output_array(value_dictionary, sample)
#   print(np.shape(output_array))
  



#   for item in sample:
#     print(value_dictionary[item])

#   training_inputs = {}
#   if not 'training_input_' + str(rate_1).replace('.', '_') + '.npy' in os.listdir('.'):
#     training_inputs[str(rate_1)] = contact_simulation
#     ''' etc. ''' 


def test16():
  rate1 = 0.01
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}
  locations = get_locations(trees)
  nodes_to_tree_dictionary = make_nodes_to_tree_dictionary(trees)
  reconstructed_locations_dictionary = make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary)
  time_depths_dictionary = make_time_depths_dictionary(trees)
  parent_dictionary = make_parent_dictionary(trees)
  contemporary_neighbour_dictionary = make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary)
  potential_donors = make_potential_donors(reconstructed_locations_dictionary, time_depths_dictionary, contemporary_neighbour_dictionary)
  trees = contact_simulation(trees, substitution_matrix, states, base_frequencies, rate1, locations, nodes_to_tree_dictionary, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary, contemporary_neighbour_dictionary, potential_donors)  
  json.dump(trees, open('tree_result.json', 'w'), indent=4)

def test17():
  rate1 = 0.01
  rate2 = 0.1
  test_two_models(rate1, rate2)

def test18():
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  make_relatedness_pairs_dictionary(list_of_languages, trees)

def test19():
  trees = make_trees()
  parent_dictionary = make_parent_dictionary(trees)
  node_1 = "'Central Western Macedonian [cent2341]':1"
  node_2 = "'Banat Bulgarian [bana1308]':1"
  x = find_phyletic_distance(node_1, node_2, parent_dictionary)
  print(x)

def test20():
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  parent_dictionary = make_parent_dictionary(trees)
  make_relatedness_pairs_dictionary(list_of_languages, trees, parent_dictionary)
  make_distance_pairs_dictionary(list_of_languages)

def test21():
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  number_of_samples = 900
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
  parent_dictionary = make_parent_dictionary(trees)
  relatedness_pairs_dictionary = make_relatedness_pairs_dictionary(list_of_languages, trees, parent_dictionary)
  relatedness_array = make_relatedness_array(list_of_languages, sample, relatedness_pairs_dictionary)
  print(np.shape(relatedness_array))
  distance_array = make_distance_array(list_of_languages, sample, relatedness_pairs_dictionary)
  print(np.shape(distance_array))

def test22():
  trees = make_trees() 
  list_of_languages = get_languages_in_grambank()  
  number_of_samples = 900
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}
  rate1 = 0.01
  input_array, output_array, relatedness_array, distance_array = make_all_arrays(trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate1, number_of_simulations=1)
  np.save('input_array.npy', input_array)
  np.save('output_array.npy', output_array)
  np.save('relatedness_array.npy', relatedness_array)
  np.save( 'distance_array.npy', distance_array)

def test23():
  trees = make_trees() 
  list_of_languages = get_languages_in_grambank()  
  number_of_samples = 900
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}
  rate1 = 0.01
  input_array, output_array, relatedness_array, distance_array = make_all_arrays(trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate1, number_of_simulations=1)
  np.save('input_array.npy', input_array)
  np.save('output_array.npy', output_array)
  np.save('relatedness_array.npy', relatedness_array)
  np.save('distance_array.npy', distance_array)

def test24():
  trees = make_trees() 
  list_of_languages = get_languages_in_grambank() 
  number_of_languages = len(list_of_languages) 
  number_of_samples = 900
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}
  rate1 = 0.1
  number_of_relatedness_bins = 10
  number_of_distance_bins = 10
  number_of_simulations = 1
  input_array, output_array, relatedness_array, distance_array = make_all_arrays(trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate1, number_of_simulations=number_of_simulations, number_of_relatedness_bins=number_of_relatedness_bins, number_of_distance_bins=number_of_distance_bins)
  np.save('input_array.npy', input_array)
  np.save('output_array.npy', output_array)
  np.save('relatedness_array.npy', relatedness_array)
  np.save('distance_array.npy', distance_array)
  na_array_1 = np.ones([1, number_of_samples, 1])
  na_array_2 = np.ones([1, 1, number_of_languages]) 
  model = Model(number_of_simulations, number_of_samples, number_of_languages, number_of_relatedness_bins, number_of_distance_bins) 
  model.train(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, steps=200)

def test_two_models(borrowing_rate_1, borrowing_rate_2):
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  number_of_samples = 900
  number_of_languages = len(list_of_languages)
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
  number_of_training_simulations = 3
  number_of_test_simulations = 3
  rate1 = borrowing_rate_1
  rate2 = borrowing_rate_2
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  states = ['0', '1']
  base_frequencies = {'0': 1, '1': 0}
  number_of_relatedness_bins = 10
  number_of_distance_bins = 10
  training_1_input, training_1_output, relatedness_array, distance_array = make_all_arrays(trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate1, number_of_simulations=number_of_training_simulations, number_of_relatedness_bins=number_of_relatedness_bins, number_of_distance_bins=number_of_distance_bins)
  na_array_1 = np.ones([1, number_of_samples, 1])
  na_array_2 = np.ones([1, 1, number_of_languages]) 
  model = Model(number_of_samples, number_of_languages, number_of_relatedness_bins, number_of_distance_bins) 
  model.train(training_1_input, training_1_output, na_array_1, na_array_2, relatedness_array, distance_array, steps=200)
  test_1_input, test_1_output = make_input_and_output_arrays(trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate1, number_of_simulations=number_of_test_simulations)
#   model.show_loss(test_1_input, test_1_output, na_array_1, na_array_2, relatedness_array, distance_array)
  training_2_input, training_2_output = make_input_and_output_arrays(trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate2, number_of_simulations=number_of_training_simulations)
  test_2_input, test_2_output = make_input_and_output_arrays(trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate2, number_of_simulations=number_of_test_simulations)
  loss_1 = model.show_loss(test_2_input, test_2_output, na_array_1, na_array_2, relatedness_array, distance_array)
  model.train(training_2_input, training_2_output, na_array_1, na_array_2, relatedness_array, distance_array, steps=200)
  loss_2 = model.show_loss(test_2_input, test_2_output, na_array_1, na_array_2, relatedness_array, distance_array)
  print('Loss 1: ', loss_1)
  print('Loss 2: ', loss_2)


#   print(np.shape(input_array))
#   print(np.shape(output_array))



def test25():
  borrowing_rate_1 = 0.01
  borrowing_rate_2 = 0.02
  test_two_models(borrowing_rate_1, borrowing_rate_2)

test25()




'''
  need a new pipeline.
  the easiest would be to prepare the average of the closest relatives and the average of 
  neighbours within 500 km.
  you have a layer of values for closest relatives, a layer for neighbours, and the intercept.
  
  another way would be to prepare an array of genealogical relatedness for the samples,
  and geographical distance.
  
  
  so the work flow should be;
  you still produce the simulated feature array.
  you have samples, which is an array of indeces.
  you then want to make an array of shape (samples, languages)
  this array is an array of genealogical distances of each member of the sample to each of the input languages.
  
  call this function 
  make_genealogical_distances_array(trees, samples, 
  
  how would i design this workflow from scratch?
  
  you have the simulation, where you assign values.  then what?
  
  you have the list of tips that are in the simulation.
  you sample from this list to make the sample of languages.
  
  you have a dictionary which is already prepared, which is the genealogical relationships between tips in the list of languages.
  you also have a dictionary of distances.
  
  so you can query these dictionaries when making the genealogical distance and geographical distance arrays.
  
  you make an array using trees and the language list; this is the values for the languages for that simulation.
  you make an array using trees and the sample list; this is the value for the sample.
  you append these arrays to input and output respectively.
  
  you also make the genealogical distance array, which are only needed once and can be broadcast.
  you have the language list and the sample list; for each language, for each member of the sample, you find
  the genealogical distance.  you end up with an array of shape (languages, samples)
  do the same for geographical distance.
  
  'unrelated' is one possible value for geographical distance.
  
  you need to change these arrays into one hot vectors next.  
  so it is turned from (languages, samples) to (languages, samples, number of bins)
  you can do all of this in numpy.
  
  then the model loads the input array of shape (None, 1, languages),


  weights was previously of shape (1, samples, languages)
  and you then multiplied element-wise by the input.  
  
  
  in the new method,
  you have input of (None, 1, languages),
  then you are trying to make a multiplier of shape (1, samples, languages)
  you make this by using the relatedness array which is of shape (1, samples, languages, number of bins),
  multiplied element-wise by relatedness-weights which are of shape (1,1,1,number of bins),
  then you reduce sum along the last axis, making
  (1, samples, languages).
  the same applies to the geographical distances array.

  
  you multiply this by the input to make
  relatedness_values of shape (None, samples, languages)
  geographical_distance_values of shape (None, samples, languages),
  and the intercept of shape [1]

  you have relateness probability a and distance probability b
  
  1 - (1-a)(1-b) is the probability of having a shared value between two languages,
  if you think of a and b as the probability of having a shared value due to relatedness and due to contact respectively
  
  then 1 - that * intercept.
  
  
  besides that, I would like the pipeline to make it easy to store training and test data, without having to write out the steps
  make a function along the lines of 
  test_two_models(borrowing_rate_1, borrowing_rate_2).
  you could make this as a test function.

  
  
  
'''



#   contact_simulation_writing_to_file('simulated_feature_array_test_002.npy', trees, list_of_languages, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair=rate2, number_of_simulations=number_of_simulations)
#   array = np.load('simulated_feature_array_test_001.npy')
#   print(np.sum(array))
#   array = np.load('simulated_feature_array_test_002.npy')
#   print(np.sum(array))
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
  







