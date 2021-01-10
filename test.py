from TreeFunctions import *
from simulation import *
import json
from preprocessing_for_grambank import *
from model import *
import os
from pipelines import *

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
def test25():
  borrowing_rate_1 = 0.01
  borrowing_rate_2 = 0.02
  test_two_models(borrowing_rate_1, borrowing_rate_2)

def test26():
  '''
  loading the grambank data
  
  
  '''
  df = readData('data.txt')
  features = getUniqueFeatures(df)
  f = getMultistateFeatures(df)
  x= getAllValues(df, 'zuni1245')
  print(x)
  x = createDataFrame(df)
  print(x)
  x = createDictionary(df)
  print(x['zuni1245']['values'])
  grambank_data = x
  
  
  '''
  how do I want to inspect it?
  you want to make the input and output arrays again;
  also need the sample.
  
  '''
  
  feature_index = 0
  print(features[feature_index])

  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  number_of_samples = 900
  number_of_languages = len(list_of_languages)
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
  number_of_relatedness_bins = 10
  number_of_distance_bins = 10
  
def test27():
  grambank_value_dictionary = get_grambank_value_dictionary()
  feature_name = 'GB131'
  value_dictionary = further_preprocessing_of_grambank_value_dictionary(grambank_value_dictionary, feature_name)
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  number_of_samples = 900
  number_of_languages = len(list_of_languages)
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
  number_of_relatedness_bins = 10
  number_of_distance_bins = 10

  input_array, output_array, relatedness_array, distance_array, na_array_1, na_array_2 = make_all_arrays_for_grambank(value_dictionary, trees, list_of_languages, sample, number_of_relatedness_bins=10, number_of_distance_bins=10)
  '''temporarily not using the na arrays:'''  
  na_array_1 = np.ones([1, number_of_samples, 1])
  na_array_2 = np.ones([1, 1, number_of_languages]) 
  model = Model(number_of_samples, number_of_languages, number_of_relatedness_bins, number_of_distance_bins) 
  model.train(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, steps=200)
  model.show_weights()


def test28():
  '''
  two goals:
  
  1. have a function which takes an input, output etc. array and finds the parameters which fit best
  2. have a way of testing this with the simulated data to show the performance of this method
  
  will do the first goal in this test
  
  the function takes 
  input_array
  output_array
  na_array_1
  na_array_2
  trees
  list_of_languages
  sample
  states
  number_of_relatedness_bins
  number_of_distance_bins
  number_of_training_simulations
  number of steps
  
  it returns a dictionary with:
  
  'substitution_matrix'
  'base_frequencies', 
  'rate_per_branch_length_per_pair'
  
  also perhaps ideally later a confidence (not for that particular analysis, but in general).
  '''
  grambank_value_dictionary = get_grambank_value_dictionary()
  feature_name = 'GB131'
  value_dictionary = further_preprocessing_of_grambank_value_dictionary(grambank_value_dictionary, feature_name)
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  states = ['0', '1']
  number_of_samples = 900
  number_of_languages = len(list_of_languages)
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
  number_of_relatedness_bins = 10
  number_of_distance_bins = 10
  number_of_simulations = 3
  number_of_steps = 120
  input_array, output_array, relatedness_array, distance_array, na_array_1, na_array_2 = make_all_arrays_for_grambank(value_dictionary, trees, list_of_languages, sample, number_of_relatedness_bins=number_of_relatedness_bins, number_of_distance_bins=number_of_distance_bins) 
  x = search_through_parameters_single_feature(input_array, output_array, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, number_of_relatedness_bins=number_of_relatedness_bins, number_of_distance_bins=number_of_distance_bins, number_of_simulations=number_of_simulations, number_of_steps=number_of_steps)  

def child_dictionary_test():
  trees = make_trees()
  child_dictionary = make_child_dictionary(trees)

@timeit
def test29():
  grambank_value_dictionary = get_grambank_value_dictionary()
  feature_name = 'GB131'
  value_dictionary = further_preprocessing_of_grambank_value_dictionary(grambank_value_dictionary, feature_name)
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  states = ['0', '1']
  number_of_samples = 900
  number_of_languages = len(list_of_languages)
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
  number_of_relatedness_bins = 10
  number_of_distance_bins = 10
  number_of_simulations = 4
  number_of_steps = 120
  input_array, output_array, relatedness_array, distance_array, na_array_1, na_array_2 = make_all_arrays_for_grambank(value_dictionary, trees, list_of_languages, sample, number_of_relatedness_bins=number_of_relatedness_bins, number_of_distance_bins=number_of_distance_bins) 
  result = search_through_parameters_single_feature(input_array, output_array, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, number_of_relatedness_bins=number_of_relatedness_bins, number_of_distance_bins=number_of_distance_bins, number_of_simulations=number_of_simulations, number_of_steps=number_of_steps)  
  print(result)

@timeit
def random_test_1():
  for i in range(100000):
    x = np.random.choice([0,1],1)

@timeit
def random_test_2():
  x = np.random.choice([0,1],100000)

@timeit
def random_test_3():
  for i in range(100000):
    x = np.random.random()

@timeit
def random_test_4():
  x = np.random.random(100000)

@timeit
def func_1(x, y, length):
  for i in range(length):
    a = x[i]
    b = y[i]

@timeit
def func_2(x,y):
  z = zip(x,y)
  for item in z:
    a = item[0]
    b = item[1]  
  

def zip_test_1():
  length = 10000000
  x = rep(1, length)
  y = rep(2, length)
  func_1(x,y,length)


def zip_test_2():
  length = 10000000
  x = rep(1, length)
  y = rep(2, length)
  func_2(x,y)


def test30():
  '''
  now want to test with simulated data
  
  so make a function for generating random values for the substitution matrix, base frequencies and borrowability,
  making the arrays, and searching through the parameters and comparing the results.s
  

  '''
  search_through_parameters_single_feature_accuracy_test()
  

'''
  
  want to begin the pipeline of testing for dependency between two features
  
  simulation needs new functions;
  
  the idea is that you create the contact events;
  you have the parameters for each feature;
  
  in one scenario, you simulate each feature independently but using the same contact events.
  
  in the other scenario, you simulate the first feature as normal;
  the other feature is simulated with the following rules:
  1. say that state A of feature 1 is linked to state B of feature 2.
  2. the substitution matrix of feature 2 is as normal, except that for that language if 
  feature 1 has state A then the transition probabilities for 
  feature 2 change: the probability of transitioning to state B increases (e.g. to 0.95), and the probability
  of transitioning away from state B decreases (e.g. to 0.05).
  e.g. say that feature 1 has matrix 
  [[0.9, 0.1], [0.1, 0.9]], and so does feature 2. say that state 1 of feature 1 and state 1 of feature 2
  are linked.  then if the language has state 1, then the matrix for feature 2 becomes
  [[0.05, 0.95], [0.05, 0.95]].
  you simulate the two features together.
  
  where should i write this?  presumably again in simulation.py.  or could make a new file,
  dependency_simulation.py.
  
  
  
  the model this time is looking at the probability of a language having a particular value given it has
  a particular value for the first feature, + some intercept for the particular value.  the model fits that probability,
  then you calculate the loss.
  
  need to first make simulation of contact event probability + borrowability as two separate things.
  
  that in itself is not difficult, but requires modifying function inputs carefully.
  ADD 'borrowability' as a variable immediately after rate per branch lenght per pair in all relevant functions.
'''
  

def test31():
  '''
  test new contact simulation
  '''
  trees = make_trees()
  substitution_matrix_list = [ [[0.95,0.05],[0.05,0.95]], [[0.9,0.1],[0.1,0.9]] ]
  states_list = [ ['0','1'], ['0', '1'] ]
  base_frequencies_list = [ {'0': 1, '1': 0}, {'0': 0.8, '1': 0.2} ]
  rate_per_branch_length_per_pair = 0.03
  borrowability_list = [1, 1]
  locations = get_locations(trees)
  nodes_to_tree_dictionary = make_nodes_to_tree_dictionary(trees)
  reconstructed_locations_dictionary = make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary)
  time_depths_dictionary = make_time_depths_dictionary(trees)
  parent_dictionary = make_parent_dictionary(trees)
  contemporary_neighbour_dictionary = make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary)
  potential_donors = make_potential_donors(reconstructed_locations_dictionary, time_depths_dictionary, contemporary_neighbour_dictionary)
  child_dictionary = make_child_dictionary(trees)  
  final_result_trees = contact_simulation(trees, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, locations, nodes_to_tree_dictionary, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary, contemporary_neighbour_dictionary, potential_donors, child_dictionary)
  json.dump(final_result_trees, open('final_result_trees.json', 'w'), indent=4)

def test32():
  trees = make_trees()
  substitution_matrix_list = [ [[0.95,0.05],[0.05,0.95]], [[0.9,0.1],[0.1,0.9]] ]
  states_list = [ ['0','1'], ['0', '1'] ]
  base_frequencies_list = [ {'0': 1, '1': 0}, {'0': 0.8, '1': 0.2} ]
  rate_per_branch_length_per_pair = 0.03
  borrowability_list = [1, 1]
  locations = get_locations(trees)
  nodes_to_tree_dictionary = make_nodes_to_tree_dictionary(trees)
  reconstructed_locations_dictionary = make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary)
  time_depths_dictionary = make_time_depths_dictionary(trees)
  parent_dictionary = make_parent_dictionary(trees)
  contemporary_neighbour_dictionary = make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary)
  potential_donors = make_potential_donors(reconstructed_locations_dictionary, time_depths_dictionary, contemporary_neighbour_dictionary)
  child_dictionary = make_child_dictionary(trees)  
  final_result_trees = contact_simulation(trees, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, locations, nodes_to_tree_dictionary, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary, contemporary_neighbour_dictionary, potential_donors, child_dictionary)
  list_of_languages = get_languages_in_grambank()  
  value_dictionary = make_value_dictionary(final_result_trees, list_of_languages) 
  print(value_dictionary)

def test33():
  '''
  now need to make input and output arrays for the new model
  input is of shape [None, 1, number_of_languages, number_of_features]
  output is of shape [None, number_of_samples, 1, number_of_features]
  
  
  '''
  trees = make_trees()
  substitution_matrix_list = [ [[0.95,0.05],[0.05,0.95]], [[0.9,0.1],[0.1,0.9]] ]
  states_list = [ ['0','1'], ['0', '1'] ]
  base_frequencies_list = [ {'0': 1, '1': 0}, {'0': 0.8, '1': 0.2} ]
  
  
  substitution_matrix_list = rep([[0.95,0.05],[0.05,0.95]], 10)
  states_list = rep(['0','1'], 10)
  base_frequencies_list = rep({'0': 0.9, '1': 0.1}, 10)
  borrowability_list = rep(1, 10)
  rate_per_branch_length_per_pair = 0.03
  
  locations = get_locations(trees)
  nodes_to_tree_dictionary = make_nodes_to_tree_dictionary(trees)
  reconstructed_locations_dictionary = make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary)
  time_depths_dictionary = make_time_depths_dictionary(trees)
  parent_dictionary = make_parent_dictionary(trees)
  contemporary_neighbour_dictionary = make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary)
  potential_donors = make_potential_donors(reconstructed_locations_dictionary, time_depths_dictionary, contemporary_neighbour_dictionary)
  child_dictionary = make_child_dictionary(trees)  
  final_result_trees = contact_simulation(trees, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, locations, nodes_to_tree_dictionary, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary, contemporary_neighbour_dictionary, potential_donors, child_dictionary)
  list_of_languages = get_languages_in_grambank()  
  value_dictionary = make_value_dictionary(final_result_trees, list_of_languages) 
  number_of_samples = 900
  number_of_languages = len(list_of_languages)
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
  print(value_dictionary)
  number_of_simulations = 1
  input_array, output_array = make_input_and_output_arrays(trees, list_of_languages, sample, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, number_of_simulations)
  print(np.shape(input_array))
  print(np.shape(output_array))

test33()






