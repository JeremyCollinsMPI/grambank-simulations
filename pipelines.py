from simulation import *
from model import *
from copy import deepcopy
from preprocessing_for_grambank import *

'''
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

actually it also needs relatedness and distance arrays

'''

SUBSTITUTION_MATRIX_0_TO_1 = 'substitution_matrix_0_to_1'
SUBSTITUTION_MATRIX_1_TO_0 = 'substitution_matrix_1_to_0'
BASE_FREQUENCIES = 'base_frequencies'
RATE_PER_BRANCH_LENGTH_PER_PAIR = 'rate_per_branch_length_per_pair'



def create_initial_substitution_matrix(states):
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  return substitution_matrix
  
def create_initial_base_frequencies(states):
  base_frequencies = {'0': 1, '1': 0}
  return base_frequencies

def create_initial_borrowing_event_rate():
  return 0.1

def propose_new_single_feature(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, trees, list_of_languages, sample, substitution_matrix, states, context, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations, number_of_steps, loss, model, proposal_rate_dictionary):
  new_substitution_matrix = deepcopy(substitution_matrix)
  new_base_frequencies = deepcopy(base_frequencies)
  new_rate_per_branch_length_per_pair = rate_per_branch_length_per_pair 
  dice_roll = np.random.choice([0, 1, 2, 3], 1)[0]
  if dice_roll == 0:
    to_change = SUBSTITUTION_MATRIX_0_TO_1
    jump = proposal_rate_dictionary[to_change]
    dice_roll_2 = np.random.choice([0, 1], 1)[0]
    if dice_roll_2 == 0:
      new_substitution_matrix[0][0] = new_substitution_matrix[0][0] + jump
      new_substitution_matrix[0][1] = new_substitution_matrix[0][1] - jump
      new_substitution_matrix[0][0] = min(1, max(new_substitution_matrix[0][0], 0))
      new_substitution_matrix[0][1] = min(1, max(new_substitution_matrix[0][1], 0))
    if dice_roll_2 == 1:
      new_substitution_matrix[0][0] = new_substitution_matrix[0][0] - jump
      new_substitution_matrix[0][1] = new_substitution_matrix[0][1] + jump
      new_substitution_matrix[0][0] = min(1, max(new_substitution_matrix[0][0], 0))
      new_substitution_matrix[0][1] = min(1, max(new_substitution_matrix[0][1], 0))
  if dice_roll == 1:
    to_change = SUBSTITUTION_MATRIX_1_TO_0
    jump = proposal_rate_dictionary[to_change]
    dice_roll_2 = np.random.choice([0, 1], 1)[0]
    if dice_roll_2 == 0:
      new_substitution_matrix[1][0] = new_substitution_matrix[1][0] + jump
      new_substitution_matrix[1][1] = new_substitution_matrix[1][1] - jump
      new_substitution_matrix[1][0] = min(1, max(new_substitution_matrix[1][0], 0))
      new_substitution_matrix[1][1] = min(1, max(new_substitution_matrix[1][1], 0))
    if dice_roll_2 == 1:
      new_substitution_matrix[1][0] = new_substitution_matrix[1][0] - jump
      new_substitution_matrix[1][1] = new_substitution_matrix[1][1] + jump
      new_substitution_matrix[1][0] = min(1, max(new_substitution_matrix[1][0], 0))
      new_substitution_matrix[1][1] = min(1, max(new_substitution_matrix[1][1], 0))
  if dice_roll == 2:
    to_change = BASE_FREQUENCIES
    jump = proposal_rate_dictionary[to_change]
    dice_roll_2 = np.random.choice([0, 1], 1)[0]
    if dice_roll_2 == 0:
      new_base_frequencies['0'] = new_base_frequencies['0'] + jump
      new_base_frequencies['1'] = new_base_frequencies['1'] - jump
      new_base_frequencies['0'] = min(1, max(new_base_frequencies['0'], 0))
      new_base_frequencies['1'] = min(1, max(new_base_frequencies['1'], 0))
    if dice_roll_2 == 1:
      new_base_frequencies['0'] = new_base_frequencies['0'] - jump
      new_base_frequencies['1'] = new_base_frequencies['1'] + jump
      new_base_frequencies['0'] = min(1, max(new_base_frequencies['0'], 0))
      new_base_frequencies['1'] = min(1, max(new_base_frequencies['1'], 0))
  if dice_roll == 3:
    to_change = RATE_PER_BRANCH_LENGTH_PER_PAIR
    jump = proposal_rate_dictionary[to_change]
    dice_roll_2 = np.random.choice([0, 1], 1)[0]
    if dice_roll_2 == 0:
      new_rate_per_branch_length_per_pair = new_rate_per_branch_length_per_pair + jump
    if dice_roll_2 == 1:
      new_rate_per_branch_length_per_pair = new_rate_per_branch_length_per_pair - jump
    new_rate_per_branch_length_per_pair = max(0, new_rate_per_branch_length_per_pair)
    print(new_rate_per_branch_length_per_pair)
  training_input, training_output = make_input_and_output_arrays(trees, list_of_languages, sample, [new_substitution_matrix], [states], [new_base_frequencies], new_rate_per_branch_length_per_pair, [1], number_of_simulations, context)    
  print('training model')
  model.train(training_input, training_output, na_array_1, na_array_2, relatedness_array, distance_array, steps=number_of_steps)
  new_loss = model.show_loss(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array)
  print(loss)
  print(new_loss)
  if new_loss < loss:
    print('accept')
    substitution_matrix = deepcopy(new_substitution_matrix)
    base_frequencies = deepcopy(new_base_frequencies)
    rate_per_branch_length_per_pair = new_rate_per_branch_length_per_pair
    loss = new_loss
    proposal_rate_dictionary[to_change] = proposal_rate_dictionary[to_change] * 1.5
  else:
    proposal_rate_dictionary[to_change] = proposal_rate_dictionary[to_change] / 1.5
  
  return substitution_matrix, base_frequencies, rate_per_branch_length_per_pair, loss, proposal_rate_dictionary

def refresh_single_feature(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations, number_of_steps, loss, model, proposal_rate_dictionary):
  training_input, training_output = make_input_and_output_arrays(trees, list_of_languages, sample, [substitution_matrix], [states], [base_frequencies], rate_per_branch_length_per_pair, [1], number_of_simulations)  
  model.train(training_input, training_output, na_array_1, na_array_2, relatedness_array, distance_array, steps=number_of_steps)
  loss = model.show_loss(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array)
  return substitution_matrix, base_frequencies, rate_per_branch_length_per_pair, loss, proposal_rate_dictionary

def search_through_parameters_single_feature(input_array, output_array, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, context, number_of_relatedness_bins, number_of_distance_bins, number_of_simulations, number_of_steps):
  substitution_matrix = create_initial_substitution_matrix(states)
  base_frequencies = create_initial_base_frequencies(states)
  rate_per_branch_length_per_pair = create_initial_borrowing_event_rate() 
  number_of_samples = len(sample)
  number_of_languages = len(list_of_languages)
  number_of_features = 1
  model = Model(number_of_samples, number_of_languages, number_of_features, number_of_relatedness_bins, number_of_distance_bins) 
  '''temporarily not using the na arrays:'''  
  na_array_1 = np.ones([1, number_of_samples, 1, number_of_features])
  na_array_2 = np.ones([1, 1, number_of_languages, number_of_features])
  loss = 1000
  proposal_rate_dictionary = {SUBSTITUTION_MATRIX_0_TO_1: 0.1, SUBSTITUTION_MATRIX_1_TO_0: 0.1, BASE_FREQUENCIES: 0.1, RATE_PER_BRANCH_LENGTH_PER_PAIR: 0.1}
  for i in range(100):
    substitution_matrix, base_frequencies, rate_per_branch_length_per_pair, loss, proposal_rate_dictionary = propose_new_single_feature(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, trees, list_of_languages, sample, substitution_matrix, states, context, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations, number_of_steps, loss, model, proposal_rate_dictionary)
    result = {'substitution_matrix': substitution_matrix, 'base_frequencies': base_frequencies, 'rate_per_branch_length_per_pair': rate_per_branch_length_per_pair}
    print(result) 
    print(proposal_rate_dictionary)
    if i % 10 == 0 and not i == 0:
      print(loss)
      refresh_number_of_simulations = 10
      substitution_matrix, base_frequencies, rate_per_branch_length_per_pair, loss, proposal_rate_dictionary = refresh_single_feature(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, refresh_number_of_simulations, number_of_steps, loss, model, proposal_rate_dictionary)      
      print('Refreshed loss: ', loss)
      proposal_rate_dictionary = {SUBSTITUTION_MATRIX_0_TO_1: 0.1, SUBSTITUTION_MATRIX_1_TO_0: 0.1, BASE_FREQUENCIES: 0.1, RATE_PER_BRANCH_LENGTH_PER_PAIR: 0.1}
      
  return result


def make_random_substitution_matrix():
  rate_1 = (np.random.random() / 2) + 0.5
  rate_2 = (np.random.random() / 2) + 0.5
  matrix = [[rate_1, 1 - rate_1], [1 - rate_2, rate_2]]
  return matrix

def make_random_base_frequencies():
  rate = (np.random.random() / 2) + 0.5
  return {'0': rate, '1': 1 - rate}

def make_random_rate_per_branch_length_per_pair():
  rate = np.random.random() / 5
  return rate
  

def make_random_borrowability():
  rate = np.random.random()
  return rate

def search_through_parameters_single_feature_accuracy_test():
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
  substitution_matrix = make_random_substitution_matrix()
  base_frequencies = make_random_base_frequencies()
  rate_per_branch_length_per_pair = make_random_rate_per_branch_length_per_pair()
  base_frequencies_list = [base_frequencies]
  states_list = [states]
  borrowability_list = [1.0]
  substitution_matrix_list = [substitution_matrix]  
  test_input, test_output, relatedness_array, distance_array = make_all_arrays(trees, list_of_languages, sample, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, number_of_simulations, number_of_relatedness_bins=10, number_of_distance_bins=10) 
  na_array_1 = np.ones([1, number_of_samples, 1])
  na_array_2 = np.ones([1, 1, number_of_languages]) 
  result = search_through_parameters_single_feature(test_input, test_output, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, number_of_relatedness_bins=number_of_relatedness_bins, number_of_distance_bins=number_of_distance_bins, number_of_simulations=number_of_simulations, number_of_steps=number_of_steps)  
  print(result)
  truth = {'substitution_matrix': substitution_matrix, 'base_frequencies': base_frequencies, 'rate_per_branch_length_per_pair': rate_per_branch_length_per_pair}
  print(truth)

def in_trees(item, trees):
  for tree in trees:
    for key in tree:
      glottocode = find_glottocode(key)      
      if item == glottocode:
        return True
  return False      

def make_reduced_list_of_languages(list_of_languages, trees):
  result = []
  for item in list_of_languages:
    if in_trees(item, trees):
      result.append(item)
  return result
        
def search_through_parameters_single_feature_sanity_check():
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()  
  states = ['0', '1']
  number_of_samples = 900
  number_of_languages = len(list_of_languages)
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
  number_of_relatedness_bins = 10
  number_of_distance_bins = 10
  number_of_simulations = 3
  number_of_steps = 60
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  base_frequencies = {'0': 1, '1': 0}
  rate_per_branch_length_per_pair = 0.03
  base_frequencies_list = [base_frequencies]
  states_list = [states]
  borrowability_list = [1.0]
  substitution_matrix_list = [substitution_matrix]  
  test_input, test_output, relatedness_array, distance_array = make_all_arrays(trees, list_of_languages, sample, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, number_of_simulations, number_of_relatedness_bins=10, number_of_distance_bins=10) 
  na_array_1 = np.ones([1, number_of_samples, 1])
  na_array_2 = np.ones([1, 1, number_of_languages]) 
  result = search_through_parameters_single_feature(test_input, test_output, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, number_of_relatedness_bins=number_of_relatedness_bins, number_of_distance_bins=number_of_distance_bins, number_of_simulations=number_of_simulations, number_of_steps=number_of_steps)  
  print(result)
  truth = {'substitution_matrix': substitution_matrix, 'base_frequencies': base_frequencies, 'rate_per_branch_length_per_pair': rate_per_branch_length_per_pair}
  print(truth)

def search_through_parameters_single_feature_sanity_check_reduced():
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()
  remake = False
  trees = make_reduced_trees(trees, list_of_languages, remake=remake)
  list_of_languages = make_reduced_list_of_languages(list_of_languages, trees)
  locations = get_locations(trees, remake=remake)
  nodes_to_tree_dictionary = make_nodes_to_tree_dictionary(trees, remake=remake)
  reconstructed_locations_dictionary = make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary, remake=remake)
  time_depths_dictionary = make_time_depths_dictionary(trees, remake=remake)
  parent_dictionary = make_parent_dictionary(trees, remake=remake)
  contemporary_neighbour_dictionary = make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary, remake=remake)
  potential_donors = make_potential_donors(reconstructed_locations_dictionary, time_depths_dictionary, contemporary_neighbour_dictionary, remake=remake)
  child_dictionary = make_child_dictionary(trees, remake=remake)  

  context = {}
  context[TREES] = trees
  context[LIST_OF_LANGUAGES] = list_of_languages
  context[LOCATIONS] = locations
  context[NODES_TO_TREE_DICTIONARY] = nodes_to_tree_dictionary
  context[RECONSTRUCTED_LOCATIONS_DICTIONARY] = reconstructed_locations_dictionary
  context[PARENT_DICTIONARY] = parent_dictionary
  context[CONTEMPORARY_NEIGHBOUR_DICTIONARY] = contemporary_neighbour_dictionary
  context[POTENTIAL_DONORS] = potential_donors
  context[CHILD_DICTIONARY] = child_dictionary
  context[TIME_DEPTHS_DICTIONARY] = time_depths_dictionary
  
  states = ['0', '1']
  number_of_samples = len(list_of_languages)
  number_of_languages = len(list_of_languages)
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
  number_of_relatedness_bins = 10
  number_of_distance_bins = 10
  number_of_simulations = 5
  '''
  maybe increase the number of simulations here
  '''
  number_of_steps = 60
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  base_frequencies = {'0': 1, '1': 0}
  rate_per_branch_length_per_pair = 0.03
  base_frequencies_list = [base_frequencies]
  states_list = [states]
  borrowability_list = [1.0]
  substitution_matrix_list = [substitution_matrix]  
  test_input, test_output, relatedness_array, distance_array = make_all_arrays(trees, list_of_languages, sample, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, number_of_simulations, context, number_of_relatedness_bins=10, number_of_distance_bins=10) 
  na_array_1 = np.ones([1, number_of_samples, 1])
  na_array_2 = np.ones([1, 1, number_of_languages]) 
  result = search_through_parameters_single_feature(test_input, test_output, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, context, number_of_relatedness_bins=number_of_relatedness_bins, number_of_distance_bins=number_of_distance_bins, number_of_simulations=number_of_simulations, number_of_steps=number_of_steps)  
  print(result)
  truth = {'substitution_matrix': substitution_matrix, 'base_frequencies': base_frequencies, 'rate_per_branch_length_per_pair': rate_per_branch_length_per_pair}
  print(truth)


'''
now want to plan a pipeline for testing whether two families had contact or are related

let's say that you have two families;
you are taking each language and comparing it to each other language.

what you could do is just take the classifier that is already trained;
and all you are changing is the relatedness dictionary.

you need to have two models trained on new data.

you can have every language;
then the model needs to have a weighting for each language, as before.

i have FamilyTestModel now.

you need to have the two families you are comparing.

you have a function for make_all_arrays_relatedness_test(), which makes the input and output array and na_arrays.

for grambank you similarly need a function make_all_arrays_for_grambank_relatedness_test.
that function currently takes a value dictionary.  

pipelines that you need:
focus first on testing whether two families are related.

1. equivalent of make all arrays, where you can specify that two families are related
2. pipeline which takes an input and output array and the two families, and tests two hypotheses, one that they are related and one that they are not
3. a pipeline for choosing two families and simulating random data for one of those hypotheses, and testing whether pipeline 2 can get it right.






'''









  
  