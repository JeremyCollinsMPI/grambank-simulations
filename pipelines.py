from simulation import *
from model import *
from copy import deepcopy

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

def propose_new(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations, number_of_steps, loss, model, proposal_rate_dictionary):
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

  training_input, training_output = make_input_and_output_arrays(trees, list_of_languages, sample, new_substitution_matrix, states, new_base_frequencies, new_rate_per_branch_length_per_pair, number_of_simulations)  
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
    proposal_rate_dictionary[to_change] = proposal_rate_dictionary[to_change] + 0.01
  else:
    proposal_rate_dictionary[to_change] = proposal_rate_dictionary[to_change] - 0.01
  
  return substitution_matrix, base_frequencies, rate_per_branch_length_per_pair, loss, proposal_rate_dictionary



def search_through_parameters_single_feature(input_array, output_array, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, number_of_relatedness_bins, number_of_distance_bins, number_of_simulations, number_of_steps):
  substitution_matrix = create_initial_substitution_matrix(states)
  base_frequencies = create_initial_base_frequencies(states)
  rate_per_branch_length_per_pair = create_initial_borrowing_event_rate() 
  number_of_samples = len(sample)
  number_of_languages = len(list_of_languages)
  model = Model(number_of_samples, number_of_languages, number_of_relatedness_bins, number_of_distance_bins) 
  '''temporarily not using the na arrays:'''  
  na_array_1 = np.ones([1, number_of_samples, 1])
  na_array_2 = np.ones([1, 1, number_of_languages]) 
  loss = 1000
  proposal_rate_dictionary = {SUBSTITUTION_MATRIX_0_TO_1: 0.1, SUBSTITUTION_MATRIX_1_TO_0: 0.1, BASE_FREQUENCIES: 0.1, RATE_PER_BRANCH_LENGTH_PER_PAIR: 0.1}
  for i in range(30):
    substitution_matrix, base_frequencies, rate_per_branch_length_per_pair, loss, proposal_rate_dictionary = propose_new(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations, number_of_steps, loss, model, proposal_rate_dictionary)
    result = {'substitution_matrix': substitution_matrix, 'base_frequencies': base_frequencies, 'rate_per_branch_length_per_pair': rate_per_branch_length_per_pair}
    print(result) 
  return result
  
  
  
  
  
  