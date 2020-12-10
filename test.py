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
  

test8()












