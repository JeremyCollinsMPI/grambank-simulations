from simulation import *
from model import *

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




def create_initial_substitution_matrix(states):
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  return substitution_matrix
  
def create_initial_base_frequencies(states):
  base_frequencies = {'0': 1, '1': 0}
  return base_frequencies


def create_initial_borrowing_event_rate():
  return 0.1

def propose_new(trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations, loss, model)
  training_input, training_output = make_input_and_output_arrays(trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations)  
  model.train(training_input, training_output, na_array_1, na_array_2, relatedness_array, distance_array, steps=number_of_steps)
  loss = model.show_loss(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array)
  
  return substitution_matrix, base_frequencies, rate_per_branch_length_per_pair, loss

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
  substitution_matrix, base_frequencies, rate_per_branch_length_per_pair, loss = propose_newpropose_new(trees, list_of_languages, sample, substitution_matrix, states, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations, loss, model)

  result = {}
  return result
  
  
  
  
  
  