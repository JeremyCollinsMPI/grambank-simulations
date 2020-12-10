from LikelihoodFunction import *
from TreeFunctions import *
import numpy as np
from general import *
from copy import deepcopy
import multiprocessing


def make_node_value(parent_value, branch_length, substitution_matrix, states):
  matrix = fractional_matrix_power(np.array(substitution_matrix), branch_length)
  parent_value_index = states.index(parent_value)
  row = matrix[parent_value_index]
  node_value_index = np.random.choice(np.arange(0, len(states)), p = row)
  return states[node_value_index]

def choose_root_value(base_frequencies):
  return np.random.choice(list(base_frequencies.keys()), p = list(base_frequencies.values()))

def assign_feature(tree, node, parent_value, substitution_matrix, states, base_frequencies):
  children = findChildren(node)
  '''
  could make this more efficient by checking the descendent tips to see whether they are included in the list of languages
  '''
  if parent_value == None:
    node_value = choose_root_value(base_frequencies)
    tree[node] = node_value
  else:
    branch_length = findBranchLength(node)
    node_value = make_node_value(parent_value, branch_length, substitution_matrix, states)
    tree[node] = node_value
  for child in children:
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
  
  
  


  
  
  
  