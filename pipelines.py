from simulation import *
from copy import deepcopy
from preprocessing_for_grambank import *

SUBSTITUTION_MATRIX_0_TO_1 = 'substitution_matrix_0_to_1'
SUBSTITUTION_MATRIX_1_TO_0 = 'substitution_matrix_1_to_0'
BASE_FREQUENCIES = 'base_frequencies'
RATE_PER_BRANCH_LENGTH_PER_PAIR = 'rate_per_branch_length_per_pair'
RELATEDNESS_PROB_SAME_ZERO = 'relatedness bins probability of second language having 0 if first language has 0' 
RELATEDNESS_PROB_SAME_ONE = 'relatedness bins probability of second language having 1 if first language has 1' 
RELATEDNESS_SAME_ZERO_ERROR = 'relatedness bins number of languages not having same value if first languages has 0'
RELATEDNESS_SAME_ONE_ERROR = 'relatedness bins number of languages not having same value if first languages has 1'
CONTACT_PROB_SAME_ZERO = 'contact bins probability of second language having 0 if first language has 0' 
CONTACT_PROB_SAME_ONE = 'contact bins probability of second language having 1 if first language has 1' 
PROPORTION_OF_ZEROS = 'proportion of zeros'

def create_initial_substitution_matrix(states):
  substitution_matrix = [[0.7, 0.3], [0.3, 0.7]]
  return substitution_matrix
  
def create_initial_base_frequencies(states):
  base_frequencies = {'0': 1, '1': 0}
  return base_frequencies

def create_initial_borrowing_event_rate():
  return 0.03

def make_summary_statistics(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array):
  '''
  changing it to use the number of languages that are incorrectly accounted for
  '''


  relatedness_total_zero = np.zeros([np.shape(relatedness_array)[3]])
  relatedness_total_one = np.zeros([np.shape(relatedness_array)[3]])
  relatedness_same_zero = np.zeros(np.shape(relatedness_total_zero))
  relatedness_same_one = np.zeros(np.shape(relatedness_total_one))
  distance_total_zero = np.zeros([np.shape(distance_array)[3]])
  distance_total_one = np.zeros([np.shape(distance_array)[3]])
  distance_same_zero = np.zeros(np.shape(distance_total_zero))
  distance_same_one = np.zeros(np.shape(distance_total_one))
  for dataset_number in range(np.shape(output_array)[0]):
    for i in range(np.shape(output_array)[1]):
      for j in range(np.shape(input_array)[2]):
        relatedness = relatedness_array[0][i][j]
        distance = distance_array[0][i][j]
        input_value = input_array[dataset_number][0][j][0]
        output_value = output_array[dataset_number][i][0][0]
        if na_array_1[dataset_number][0][j][0] == 1 and na_array_2[dataset_number][i][0][0] == 1:
          if input_value == 0:
            relatedness_total_zero = relatedness_total_zero + relatedness
            distance_total_zero = distance_total_zero + distance
            if output_value == 0:
              relatedness_same_zero = relatedness_same_zero + relatedness
              distance_same_zero = distance_same_zero + distance
          if input_value == 1:
            relatedness_total_one = relatedness_total_one + relatedness
            distance_total_one = distance_total_one + distance
            if output_value == 1:
              relatedness_same_one = relatedness_same_one + relatedness
              distance_same_one = distance_same_one + distance
  number_of_values = 0
  number_of_zeros = 0
  for dataset_number in range(np.shape(output_array)[0]):
    for j in range(np.shape(input_array)[2]):
      number_of_values = number_of_values + 1
      if input_array[dataset_number][0][j][0] == 0:
        number_of_zeros = number_of_zeros + 1
  relatedness_total_zero = np.maximum(relatedness_total_zero, 1)
  relatedness_total_one = np.maximum(relatedness_total_one, 1)
  distance_total_zero = np.maximum(distance_total_zero, 1)
  distance_total_one = np.maximum(distance_total_one, 1)
  summary_statistics = {}
#   summary_statistics[RELATEDNESS_PROB_SAME_ZERO] = relatedness_same_zero / relatedness_total_zero
#   summary_statistics[RELATEDNESS_PROB_SAME_ONE] = relatedness_same_one / relatedness_total_one
  summary_statistics[CONTACT_PROB_SAME_ZERO] = distance_same_zero / distance_total_zero
  summary_statistics[CONTACT_PROB_SAME_ONE] = distance_same_one / distance_total_one  
  summary_statistics[PROPORTION_OF_ZEROS] = number_of_zeros / number_of_values
  summary_statistics[RELATEDNESS_SAME_ZERO_ERROR] = (relatedness_total_zero - relatedness_same_zero) / np.shape(output_array)[0]
  summary_statistics[RELATEDNESS_SAME_ONE_ERROR] = (relatedness_total_one - relatedness_same_one) / np.shape(output_array)[0]
  print('Number of simulations: ', np.shape(output_array)[0])
  return summary_statistics

def find_loss(training_summary_statistics, real_summary_statistics):
  total = 0
  for x in [RELATEDNESS_SAME_ZERO_ERROR, RELATEDNESS_SAME_ONE_ERROR, CONTACT_PROB_SAME_ZERO, CONTACT_PROB_SAME_ONE, PROPORTION_OF_ZEROS]:
    total = total + np.sum(abs(training_summary_statistics[x] - real_summary_statistics[x]))
#   print('Loss: ')
  return total

def make_scheduler():
  scheduler = {'substitution_matrix_0_to_1': 
  {'adjustment': 0.1,
  'last_direction': None
  }, 'substitution_matrix_1_to_0':
  {'adjustment': 0.1,
  'last_direction': None
  }, 'base_frequencies':
  {'adjustment': 0.1,
  'last_direction': None
  }, 'rate_per_branch_length_per_pair':
  { 'adjustment': 0.1,
  'last_direction': None
  }  
  }
  return scheduler

def update_substitution_matrix(parameter_context, training_summary_statistics, real_summary_statistics, scheduler):
  use_up_to_index_number = 2
  print('Training zero: ', training_summary_statistics[RELATEDNESS_SAME_ZERO_ERROR])
  print('Real zero: ', real_summary_statistics[RELATEDNESS_SAME_ZERO_ERROR])
  print('Training one: ', training_summary_statistics[RELATEDNESS_SAME_ONE_ERROR])
  print('Real one: ', real_summary_statistics[RELATEDNESS_SAME_ONE_ERROR])
  error = training_summary_statistics[RELATEDNESS_SAME_ZERO_ERROR][0:use_up_to_index_number] - real_summary_statistics[RELATEDNESS_SAME_ZERO_ERROR][0:use_up_to_index_number]
  print('Zero error: ', np.sum(error))
  print('Previous matrix: ', parameter_context['substitution_matrix'])
  adjustment = scheduler['substitution_matrix_0_to_1']['adjustment']
  if np.sum(error) < 0:
    parameter_context['substitution_matrix'][0][0] = parameter_context['substitution_matrix'][0][0] - adjustment
    parameter_context['substitution_matrix'][0][0] = min(0.99, max(0.01, parameter_context['substitution_matrix'][0][0]))
    parameter_context['substitution_matrix'][0][1] = 1 - parameter_context['substitution_matrix'][0][0]
    current_direction = 'DOWN'
  if np.sum(error) > 0:
    parameter_context['substitution_matrix'][0][0] = parameter_context['substitution_matrix'][0][0] + adjustment
    parameter_context['substitution_matrix'][0][0] = min(0.99, max(0.01, parameter_context['substitution_matrix'][0][0]))
    parameter_context['substitution_matrix'][0][1] = 1 - parameter_context['substitution_matrix'][0][0]
    current_direction = 'UP'
  if scheduler['substitution_matrix_0_to_1']['last_direction'] == None:
    pass
  elif not current_direction == scheduler['substitution_matrix_0_to_1']['last_direction']:
    scheduler['substitution_matrix_0_to_1']['adjustment'] = max(0.01, scheduler['substitution_matrix_0_to_1']['adjustment'] - 0.02)
  scheduler['substitution_matrix_0_to_1']['last_direction'] = current_direction
  error = training_summary_statistics[RELATEDNESS_SAME_ONE_ERROR][0:use_up_to_index_number] - real_summary_statistics[RELATEDNESS_SAME_ONE_ERROR][0:use_up_to_index_number]
  print('One error: ', np.sum(error))
  adjustment = scheduler['substitution_matrix_1_to_0']['adjustment']
  if np.sum(error) < 0:
    parameter_context['substitution_matrix'][1][1] = parameter_context['substitution_matrix'][1][1] - adjustment
    parameter_context['substitution_matrix'][1][1] = min(0.99, max(0.01, parameter_context['substitution_matrix'][1][1]))
    parameter_context['substitution_matrix'][1][0] = 1 - parameter_context['substitution_matrix'][1][1]
    current_direction = 'DOWN'
  if np.sum(error) > 0:
    parameter_context['substitution_matrix'][1][1] = parameter_context['substitution_matrix'][1][1] + adjustment
    parameter_context['substitution_matrix'][1][1] = min(0.99, max(0.01, parameter_context['substitution_matrix'][1][1]))
    parameter_context['substitution_matrix'][1][0] = 1 - parameter_context['substitution_matrix'][1][1]
    current_direction = 'UP'
  if scheduler['substitution_matrix_1_to_0']['last_direction'] == None:
    pass
  elif not current_direction == scheduler['substitution_matrix_1_to_0']['last_direction']:
    scheduler['substitution_matrix_1_to_0']['adjustment'] = max(0.01, scheduler['substitution_matrix_1_to_0']['adjustment'] - 0.02)
  scheduler['substitution_matrix_1_to_0']['last_direction'] = current_direction
  print('New matrix: ', parameter_context['substitution_matrix'])
  print('Scheduler: ', scheduler)
  return parameter_context, scheduler

def update_base_frequencies(parameter_context, training_summary_statistics, real_summary_statistics, scheduler):
  print('Training proportion: ', training_summary_statistics[PROPORTION_OF_ZEROS])
  print('Real proportion: ', real_summary_statistics[PROPORTION_OF_ZEROS])
  print('Base frequencies: ', parameter_context[BASE_FREQUENCIES])
  adjustment = scheduler['base_frequencies']['adjustment']
  error = training_summary_statistics[PROPORTION_OF_ZEROS] - real_summary_statistics[PROPORTION_OF_ZEROS]
  if error > 0:
    parameter_context[BASE_FREQUENCIES]['0'] = parameter_context[BASE_FREQUENCIES]['0'] - adjustment
    parameter_context[BASE_FREQUENCIES]['0'] = min(0.99, max(0.01, parameter_context[BASE_FREQUENCIES]['0']))
    parameter_context[BASE_FREQUENCIES]['1'] = 1 - parameter_context[BASE_FREQUENCIES]['0']
    current_direction = 'DOWN'
  elif error < 0:
    parameter_context[BASE_FREQUENCIES]['0'] = parameter_context[BASE_FREQUENCIES]['0'] + adjustment
    parameter_context[BASE_FREQUENCIES]['0'] = min(0.99, max(0.01, parameter_context[BASE_FREQUENCIES]['0']))
    parameter_context[BASE_FREQUENCIES]['1'] = 1 - parameter_context[BASE_FREQUENCIES]['0']
    current_direction = 'UP'
  print('New base frequencies: ', parameter_context[BASE_FREQUENCIES])
  if scheduler['base_frequencies']['last_direction'] == None:
    pass
  elif not scheduler['base_frequencies']['last_direction'] == current_direction:
    scheduler['base_frequencies']['adjustment'] = max(0.01, scheduler['base_frequencies']['adjustment'] - 0.02)
  scheduler['base_frequencies']['last_direction'] = current_direction
  print('Scheduler: ', scheduler)
  return parameter_context, scheduler

def update_rate_per_branch_length_per_pair(parameter_context, training_summary_statistics, real_summary_statistics, scheduler):
  adjustment = scheduler['rate_per_branch_length_per_pair']['adjustment']
  error = training_summary_statistics[CONTACT_PROB_SAME_ZERO] - real_summary_statistics[CONTACT_PROB_SAME_ZERO]
  error = error + training_summary_statistics[CONTACT_PROB_SAME_ONE] - real_summary_statistics[CONTACT_PROB_SAME_ONE]
  if np.sum(error) > 0:
    parameter_context['rate_per_branch_length_per_pair'] = parameter_context['rate_per_branch_length_per_pair'] + adjustment
    parameter_context['rate_per_branch_length_per_pair'] = max(0, min(0.99, parameter_context['rate_per_branch_length_per_pair']))
    current_direction = 'UP'
  if np.sum(error) < 0:
    parameter_context['rate_per_branch_length_per_pair'] = parameter_context['rate_per_branch_length_per_pair'] - adjustment
    parameter_context['rate_per_branch_length_per_pair'] = max(0, min(0.99, parameter_context['rate_per_branch_length_per_pair']))
    current_direction = 'DOWN'
  if scheduler['rate_per_branch_length_per_pair']['last_direction'] == None:
    pass
  elif not scheduler['rate_per_branch_length_per_pair']['last_direction'] == current_direction:
    scheduler['rate_per_branch_length_per_pair']['adjustment']
  print('Rate per branch length per pair: ', parameter_context['rate_per_branch_length_per_pair'])
  print('Scheduler: ', scheduler)
  return parameter_context, scheduler

def update_parameters(parameter_context, training_summary_statistics, real_summary_statistics, scheduler):
  '''
  so you are adjusting parameters in the direction worked out by comparing the summary statistics
  
  parameter_context= update_substitution_matrix(parameter_context, training_summary_statistics, real_summary_statistics)
  parameter_context = update_base_frequencies(parameter_context, training_summary_statistics, real_summary_statistics)
  parameter_context = update_rate_per_branch_length_per_pair(parameter_context, training_summary_statistics, real_summary_statistics)
  return parameter_
  
  
  
  
  
  '''
  parameter_context, scheduler = update_substitution_matrix(parameter_context, training_summary_statistics, real_summary_statistics, scheduler)
  parameter_context, scheduler = update_base_frequencies(parameter_context, training_summary_statistics, real_summary_statistics, scheduler)
#   parameter_context, scheduler = update_rate_per_branch_length_per_pair(parameter_context, training_summary_statistics, real_summary_statistics)
  return parameter_context, scheduler

def propose_new_single_feature(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, trees, list_of_languages, sample, parameter_context, states, context, number_of_simulations, scheduler):
  new_substitution_matrix = deepcopy(parameter_context['substitution_matrix'])
  new_base_frequencies = deepcopy(parameter_context['base_frequencies'])
  new_rate_per_branch_length_per_pair = parameter_context['rate_per_branch_length_per_pair']
  training_input, training_output = make_input_and_output_arrays(trees, list_of_languages, sample, [new_substitution_matrix], [states], [new_base_frequencies], new_rate_per_branch_length_per_pair, [1], number_of_simulations, context)    
  training_summary_statistics = make_summary_statistics(training_input, training_output, na_array_1, na_array_2, relatedness_array, distance_array)
  real_summary_statistics = context['real_summary_statistics']
  loss = find_loss(training_summary_statistics, real_summary_statistics)
  parameter_context, scheduler = update_parameters(parameter_context, training_summary_statistics, real_summary_statistics, scheduler)
  return parameter_context, loss

def search_through_parameters_single_feature(input_array, output_array, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, context, number_of_relatedness_bins, number_of_distance_bins, number_of_simulations):
  substitution_matrix = create_initial_substitution_matrix(states)
  base_frequencies = create_initial_base_frequencies(states)
  rate_per_branch_length_per_pair = create_initial_borrowing_event_rate() 
  number_of_samples = len(sample)
  number_of_languages = len(list_of_languages)
  number_of_features = 1
  na_array_1 = np.ones([number_of_simulations, 1, number_of_languages, 1]) 
  na_array_2 = np.ones([number_of_simulations, number_of_samples, 1, 1])
  loss = 1000
  proposal_rate_dictionary = {SUBSTITUTION_MATRIX_0_TO_1: 0.1, SUBSTITUTION_MATRIX_1_TO_0: 0.1, BASE_FREQUENCIES: 0.1, RATE_PER_BRANCH_LENGTH_PER_PAIR: 0.1}
  context['real_summary_statistics'] = make_summary_statistics(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array)
  parameter_context = {'substitution_matrix': substitution_matrix, 'rate_per_branch_length_per_pair': rate_per_branch_length_per_pair, 'base_frequencies': base_frequencies}
  scheduler = make_scheduler()
  for i in range(20):
    context['step'] = i
    parameter_context, loss = propose_new_single_feature(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, trees, list_of_languages, sample, parameter_context, states, context, number_of_simulations, scheduler)
  result = parameter_context
  return result

def in_trees(item, trees):
  for tree in trees:
    for key in tree:
      glottocode = find_glottocode(key)      
      if item == glottocode:
        return True
  return False  

def make_reduced_list_of_languages(list_of_languages, trees, remake=True):
  if not remake:
    if 'reduced_list_of_languages.json' in os.listdir('.'):
      return json.load(open('reduced_list_of_languages.json', 'r'))
  result = []
  for item in list_of_languages:
    if in_trees(item, trees):
      result.append(item)
  json.dump(result, open('reduced_list_of_languages.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
  return result

def search_through_parameters_single_feature_sanity_check_reduced():
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()
  remake = False
  trees = make_reduced_trees(trees, list_of_languages, remake=remake)
  list_of_languages = make_reduced_list_of_languages(list_of_languages, trees, remake=remake)
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
#   number_of_samples = len(list_of_languages)
  number_of_samples = 400
  number_of_languages = len(list_of_languages)
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
  number_of_relatedness_bins = 10
  number_of_distance_bins = 10
  number_of_simulations = 10
  number_of_steps = 150
  substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
  base_frequencies = {'0': 1, '1': 0}
  rate_per_branch_length_per_pair = 0.03
  base_frequencies_list = [base_frequencies]
  states_list = [states]
  borrowability_list = [1.0]
  substitution_matrix_list = [substitution_matrix]  
  test_input, test_output, relatedness_array, distance_array = make_all_arrays(trees, list_of_languages, sample, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, 1, context, number_of_relatedness_bins=10, number_of_distance_bins=10) 
  na_array_1 = np.ones([1, 1, number_of_languages, 1]) 
  na_array_2 = np.ones([1, number_of_samples, 1, 1])
  result = search_through_parameters_single_feature(test_input, test_output, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, context, number_of_relatedness_bins=number_of_relatedness_bins, number_of_distance_bins=number_of_distance_bins, number_of_simulations=number_of_simulations)  
  print(result)
  truth = {'substitution_matrix': substitution_matrix, 'base_frequencies': base_frequencies, 'rate_per_branch_length_per_pair': rate_per_branch_length_per_pair}
  print(truth)

def main_simulation_test():
  '''
  you want to ask given a particular set of val
  '''
  test_substitution_matrices = []
  test_substitution_matrices.append([[0.95, 0.05], [0.05, 0.95]]) 
  test_rates_per_branch_length_per_pair = [0.03]
  test_base_frequencies = [{'0': 0.0, '1': 1.0}]
  runs = 1
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()
  remake = False
  trees = make_reduced_trees(trees, list_of_languages, remake=remake)
  list_of_languages = make_reduced_list_of_languages(list_of_languages, trees, remake=remake)
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
#   number_of_samples = len(list_of_languages)
  number_of_samples = 400
  number_of_languages = len(list_of_languages)
  number_of_relatedness_bins = 10  
  number_of_distance_bins = 10
  number_of_simulations = 10
  number_of_steps = 150
  results = []
  for i in range(runs):
    random_index = np.random.choice(range(len(test_substitution_matrices)), 1)[0]
    substitution_matrix = test_substitution_matrices[random_index]
    rate_per_branch_length_per_pair = test_rates_per_branch_length_per_pair[random_index]
    base_frequencies = test_base_frequencies[random_index]
    substitution_matrix_list = [substitution_matrix]
    base_frequencies_list = [base_frequencies]
    states_list = [states]
    borrowability_list = [1.0]
    sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
    test_input, test_output, relatedness_array, distance_array = make_all_arrays(trees, list_of_languages, sample, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, 1, context, number_of_relatedness_bins=10, number_of_distance_bins=10) 
    na_array_1 = np.ones([1, 1, number_of_languages, 1]) 
    na_array_2 = np.ones([1, number_of_samples, 1, 1])

    result = search_through_parameters_single_feature(test_input, test_output, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, context, number_of_relatedness_bins, number_of_distance_bins, number_of_simulations)
    print(result)
    results.append({'estimated parameters': deepcopy(result), 'true parameters': {'substitution matrix': deepcopy(substitution_matrix), 'base_frequencies': deepcopy(base_frequencies), RATE_PER_BRANCH_LENGTH_PER_PAIR: rate_per_branch_length_per_pair}})
  json.dump(results, open('main_simulation_test_results.json', 'w'))
  '''
  then want to aggregate the results in some way
  how do you want to show the result?
  just append the results.
  '''


def real_single_feature_evaluation(feature_id):
  '''
  load data for a feature in grambank and find the parameters that fit it best
  '''
  trees = make_trees()
  list_of_languages = get_languages_in_grambank()
  remake = False
  trees = make_reduced_trees(trees, list_of_languages, remake=remake)
  list_of_languages = make_reduced_list_of_languages(list_of_languages, trees, remake=remake)
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
#   number_of_samples = len(list_of_languages)
  number_of_samples = 400
  number_of_languages = len(list_of_languages)
  number_of_relatedness_bins = 10  
  number_of_distance_bins = 10
  number_of_simulations = 10
  number_of_steps = 150
  grambank_value_dictionary = get_grambank_value_dictionary()
  feature_name = 'GB131'
  value_dictionary = further_preprocessing_of_grambank_value_dictionary(grambank_value_dictionary, feature_name)
  sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)

  print(value_dictionary)

  '''
  check that next part is working
  '''
  input_array, output_array, relatedness_array, distance_array, na_array_1, na_array_2 = make_all_arrays_for_grambank(value_dictionary, trees, list_of_languages, sample, number_of_relatedness_bins=number_of_relatedness_bins, number_of_distance_bins=number_of_distance_bins)
  print(input_array)  

  na_array_1 = np.ones([1, 1, number_of_languages, 1]) 
  na_array_2 = np.ones([1, number_of_samples, 1, 1])

  result = search_through_parameters_single_feature(input_array, output_array, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, context, number_of_relatedness_bins, number_of_distance_bins, number_of_simulations)
  print(result)





















































# ----------------------
# '''
#   the function takes 
#   input_array
#   output_array
#   na_array_1
#   na_array_2
#   trees
#   list_of_languages
#   sample
#   states
#   number_of_relatedness_bins
#   number_of_distance_bins
#   number_of_training_simulations
#   number of steps
#   
#   it returns a dictionary with:
#   
#   'substitution_matrix'
#   'base_frequencies', 
#   'rate_per_branch_length_per_pair'
# 
# actually it also needs relatedness and distance arrays
# 
# '''
# 
# SUBSTITUTION_MATRIX_0_TO_1 = 'substitution_matrix_0_to_1'
# SUBSTITUTION_MATRIX_1_TO_0 = 'substitution_matrix_1_to_0'
# BASE_FREQUENCIES = 'base_frequencies'
# RATE_PER_BRANCH_LENGTH_PER_PAIR = 'rate_per_branch_length_per_pair'
# RELATEDNESS_PROB_SAME_ZERO = 'relatedness bins probability of second language having 0 if first language has 0' 
# RELATEDNESS_PROB_SAME_ONE = 'relatedness bins probability of second language having 1 if first language has 1' 
# CONTACT_PROB_SAME_ZERO = 'contact bins probability of second language having 0 if first language has 0' 
# CONTACT_PROB_SAME_ONE = 'contact bins probability of second language having 1 if first language has 1' 
# PROPORTION_OF_ZEROS = 'proportion of zeros'
# 
# def create_initial_substitution_matrix(states):
#   substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
#   return substitution_matrix
#   
# def create_initial_base_frequencies(states):
#   base_frequencies = {'0': 1, '1': 0}
#   return base_frequencies
# 
# def create_initial_borrowing_event_rate():
#   return 0.1
# 
# def make_summary_statistics(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array):
#   '''
#   you have each pair of languages;
#   you have the relatedness and the distance;
#   
#   so you can calculate the probability of two languages with a particular relatedness bin having the same value
#   it is actually two numbers:
#   the probability of the second language having 0 if the first language has 0;
#   and the probability of the second language having 1 if the first language has 1;
#   
#   similarly for contact bin.
#   
#   you also may want a summary statistic for the proportion of 0s in the dataset.  
#   
#   how do you want to structure the summary statistics?
#   
#   i will write the other functions first to make that clear
#   
#   but you will basically have
#   
#   summary_statistics as a dictionary, with keys
#   'relatedness bins probability of second language having 0 if first language has 0' RELATEDNESS_PROB_SAME_ZERO
#   'relatedness bins probability of second language having 1 if first language has 1' RELATEDNESS_PROB_SAME_ONE
#   'contact bins probability of second language having 0 if first language has 0' CONTACT_PROB_SAME_ZERO
#   'contact bins probability of second language having 1 if first language has 1' CONTACT_PROB_SAME_ONE
#   
#   PROPORTION_OF_ZEROS = 'proportion of zeros'
#   
#   
#   
#   
#   '''
# 
# 
# def update_parameters(parameter_context, training_summary_statistics, real_summary_statistics):
#   '''
#   so you are adjusting parameters in the direction worked out by comparing the summary statistics
#   
#   parameter_context= update_substitution_matrix(parameter_context, training_summary_statistics, real_summary_statistics)
#   parameter_context = update_base_frequencies(parameter_context, training_summary_statistics, real_summary_statistics)
#   parameter_context = update_rate_per_branch_length_per_pair(parameter_context, training_summary_statistics, real_summary_statistics)
#   return parameter_
#   
#   
#   
#   
#   
#   '''
# 
# 
# def propose_new_single_feature(parameter_context, 
# 
# 
# 
# 
# input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, trees, list_of_languages, sample, substitution_matrix, states, context, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations, number_of_steps, loss, model, proposal_rate_dictionary):
# 
# 
# 
# 
# 
# def propose_new_single_feature(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, trees, list_of_languages, sample, substitution_matrix, states, context, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations, number_of_steps, loss, model, proposal_rate_dictionary):
#   '''
#   new way of doing this
#   
#   you produce data using the parameters given
#   
#   you produce summary statistics for that data
#   
#   you compare it with the real data
#   
#   then update the parameters
#   
#   
#   
#   
#   '''
# 
#   new_substitution_matrix = deepcopy(substitution_matrix)
#   new_base_frequencies = deepcopy(base_frequencies)
#   new_rate_per_branch_length_per_pair = rate_per_branch_length_per_pair 
#   training_input, training_output = make_input_and_output_arrays(trees, list_of_languages, sample, [new_substitution_matrix], [states], [new_base_frequencies], new_rate_per_branch_length_per_pair, [1], number_of_simulations, context)    
#   training_summary_statistics = make_summary_statistics(training_input, training_output, na_array_1, na_array_2, relatedness_array, distance_array)
#   real_summary_statistics = context['real_summary_statistics']
#   parameter_context = {'new_substitution_matrix': new_substitution_matrix, 
#   'new_base_frequencies': new_base_frequencies, 
#   'new_rate_per_branch_length_per_pair': new_rate_per_branch_length_per_pair}
#   parameter_context = update_parameters(parameter_context, training_summary_statistics, real_summary_statistics)
#   new_substitution_matrix, new_base_frequencies, new_rate_per_branch_length_per_pair = parameter_context[
#   
#   '''
#   
#   working on this part now
#   
#   you want to update the parameters.
#   something like
#   
#     return new_substitution_matrix, new_base_frequencies, new_rate_per_branch_length_per_pair
#     
#     possibly also a proposal_rate_dictionary.
#     
#     redoing this to make the function take parameter_context as an input
#     
#   
#   '''
#   
#   
#   
# 
# 
# #   dice_roll = np.random.choice([0, 1, 2, 3], 1)[0]
# #   if dice_roll == 0:
# #     to_change = SUBSTITUTION_MATRIX_0_TO_1
# #     jump = proposal_rate_dictionary[to_change]
# #     dice_roll_2 = np.random.choice([0, 1], 1)[0]
# #     if dice_roll_2 == 0:
# #       new_substitution_matrix[0][0] = new_substitution_matrix[0][0] + jump
# #       new_substitution_matrix[0][1] = new_substitution_matrix[0][1] - jump
# #       new_substitution_matrix[0][0] = min(1, max(new_substitution_matrix[0][0], 0))
# #       new_substitution_matrix[0][1] = min(1, max(new_substitution_matrix[0][1], 0))
# #     if dice_roll_2 == 1:
# #       new_substitution_matrix[0][0] = new_substitution_matrix[0][0] - jump
# #       new_substitution_matrix[0][1] = new_substitution_matrix[0][1] + jump
# #       new_substitution_matrix[0][0] = min(1, max(new_substitution_matrix[0][0], 0))
# #       new_substitution_matrix[0][1] = min(1, max(new_substitution_matrix[0][1], 0))
# #   if dice_roll == 1:
# #     to_change = SUBSTITUTION_MATRIX_1_TO_0
# #     jump = proposal_rate_dictionary[to_change]
# #     dice_roll_2 = np.random.choice([0, 1], 1)[0]
# #     if dice_roll_2 == 0:
# #       new_substitution_matrix[1][0] = new_substitution_matrix[1][0] + jump
# #       new_substitution_matrix[1][1] = new_substitution_matrix[1][1] - jump
# #       new_substitution_matrix[1][0] = min(1, max(new_substitution_matrix[1][0], 0))
# #       new_substitution_matrix[1][1] = min(1, max(new_substitution_matrix[1][1], 0))
# #     if dice_roll_2 == 1:
# #       new_substitution_matrix[1][0] = new_substitution_matrix[1][0] - jump
# #       new_substitution_matrix[1][1] = new_substitution_matrix[1][1] + jump
# #       new_substitution_matrix[1][0] = min(1, max(new_substitution_matrix[1][0], 0))
# #       new_substitution_matrix[1][1] = min(1, max(new_substitution_matrix[1][1], 0))
# #   if dice_roll == 2:
# #     to_change = BASE_FREQUENCIES
# #     jump = proposal_rate_dictionary[to_change]
# #     dice_roll_2 = np.random.choice([0, 1], 1)[0]
# #     if dice_roll_2 == 0:
# #       new_base_frequencies['0'] = new_base_frequencies['0'] + jump
# #       new_base_frequencies['1'] = new_base_frequencies['1'] - jump
# #       new_base_frequencies['0'] = min(1, max(new_base_frequencies['0'], 0))
# #       new_base_frequencies['1'] = min(1, max(new_base_frequencies['1'], 0))
# #     if dice_roll_2 == 1:
# #       new_base_frequencies['0'] = new_base_frequencies['0'] - jump
# #       new_base_frequencies['1'] = new_base_frequencies['1'] + jump
# #       new_base_frequencies['0'] = min(1, max(new_base_frequencies['0'], 0))
# #       new_base_frequencies['1'] = min(1, max(new_base_frequencies['1'], 0))
# #   if dice_roll == 3:
# #     to_change = RATE_PER_BRANCH_LENGTH_PER_PAIR
# #     jump = proposal_rate_dictionary[to_change]
# #     dice_roll_2 = np.random.choice([0, 1], 1)[0]
# #     if dice_roll_2 == 0:
# #       new_rate_per_branch_length_per_pair = new_rate_per_branch_length_per_pair + jump
# #     if dice_roll_2 == 1:
# #       new_rate_per_branch_length_per_pair = new_rate_per_branch_length_per_pair - jump
# #     new_rate_per_branch_length_per_pair = max(0, new_rate_per_branch_length_per_pair)
# #     print(new_rate_per_branch_length_per_pair)
# 
#   print('training model')
#   if context['step'] == 0:
#     initialise=True
#   else:
#     initialise=False
#     number_of_steps=150
#   model.train(training_input, training_output, na_array_1, na_array_2, relatedness_array, distance_array, steps=number_of_steps, initialise=initialise)
#   new_loss = model.show_loss(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array)
#   print(loss)
#   print(new_loss)
#   if new_loss < loss:
#     print('accept')
#     substitution_matrix = deepcopy(new_substitution_matrix)
#     base_frequencies = deepcopy(new_base_frequencies)
#     rate_per_branch_length_per_pair = new_rate_per_branch_length_per_pair
#     loss = new_loss
#     proposal_rate_dictionary[to_change] = proposal_rate_dictionary[to_change] * 1.5
#   else:
#     proposal_rate_dictionary[to_change] = proposal_rate_dictionary[to_change] / 1.5
#   
#   return substitution_matrix, base_frequencies, rate_per_branch_length_per_pair, loss, proposal_rate_dictionary
# 
# def refresh_single_feature(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, trees, list_of_languages, sample, substitution_matrix, states, context, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations, number_of_steps, loss, model, proposal_rate_dictionary):
#   training_input, training_output = make_input_and_output_arrays(trees, list_of_languages, sample, [substitution_matrix], [states], [base_frequencies], rate_per_branch_length_per_pair, [1], number_of_simulations, context)  
#   model.train(training_input, training_output, na_array_1, na_array_2, relatedness_array, distance_array, steps=number_of_steps)
#   loss = model.show_loss(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array)
#   return substitution_matrix, base_frequencies, rate_per_branch_length_per_pair, loss, proposal_rate_dictionary
# 
# def search_through_parameters_single_feature(input_array, output_array, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, context, number_of_relatedness_bins, number_of_distance_bins, number_of_simulations, number_of_steps):
#   substitution_matrix = create_initial_substitution_matrix(states)
#   base_frequencies = create_initial_base_frequencies(states)
#   rate_per_branch_length_per_pair = create_initial_borrowing_event_rate() 
#   number_of_samples = len(sample)
#   number_of_languages = len(list_of_languages)
#   number_of_features = 1
#   model = Model(number_of_samples, number_of_languages, number_of_features, number_of_relatedness_bins, number_of_distance_bins) 
#   '''temporarily not using the na arrays:'''  
#   na_array_1 = np.ones([1, number_of_samples, 1, number_of_features])
#   na_array_2 = np.ones([1, 1, number_of_languages, number_of_features])
#   loss = 1000
#   proposal_rate_dictionary = {SUBSTITUTION_MATRIX_0_TO_1: 0.1, SUBSTITUTION_MATRIX_1_TO_0: 0.1, BASE_FREQUENCIES: 0.1, RATE_PER_BRANCH_LENGTH_PER_PAIR: 0.1}
#   context['real summary statistics'] = make_summary_statistics(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array)
#   for i in range(1):
#     context['step'] = i
#     '''
#     you could use parameter_context here
#     '''
#     par
#     
#     
#     
#     substitution_matrix, base_frequencies, rate_per_branch_length_per_pair, loss, proposal_rate_dictionary = propose_new_single_feature(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, trees, list_of_languages, sample, substitution_matrix, states, context, base_frequencies, rate_per_branch_length_per_pair, number_of_simulations, number_of_steps, loss, model, proposal_rate_dictionary)
# 
# #     result = {'substitution_matrix': substitution_matrix, 'base_frequencies': base_frequencies, 'rate_per_branch_length_per_pair': rate_per_branch_length_per_pair}
# #     print(result) 
# #     print(proposal_rate_dictionary)
# #     if i % 10 == 0 and not i == 0:
# #       print(loss)
# #       refresh_number_of_simulations = 20
# #       substitution_matrix, base_frequencies, rate_per_branch_length_per_pair, loss, proposal_rate_dictionary = refresh_single_feature(input_array, output_array, na_array_1, na_array_2, relatedness_array, distance_array, trees, list_of_languages, sample, substitution_matrix, states, context, base_frequencies, rate_per_branch_length_per_pair, refresh_number_of_simulations, number_of_steps, loss, model, proposal_rate_dictionary)      
# #       print('Refreshed loss: ', loss)
# #       proposal_rate_dictionary = {SUBSTITUTION_MATRIX_0_TO_1: 0.1, SUBSTITUTION_MATRIX_1_TO_0: 0.1, BASE_FREQUENCIES: 0.1, RATE_PER_BRANCH_LENGTH_PER_PAIR: 0.1}
#       
#   return result
# 
# 
# def make_random_substitution_matrix():
#   rate_1 = (np.random.random() / 2) + 0.5
#   rate_2 = (np.random.random() / 2) + 0.5
#   matrix = [[rate_1, 1 - rate_1], [1 - rate_2, rate_2]]
#   return matrix
# 
# def make_random_base_frequencies():
#   rate = (np.random.random() / 2) + 0.5
#   return {'0': rate, '1': 1 - rate}
# 
# def make_random_rate_per_branch_length_per_pair():
#   rate = np.random.random() / 5
#   return rate
#   
# 
# def make_random_borrowability():
#   rate = np.random.random()
#   return rate
# 
# def search_through_parameters_single_feature_accuracy_test():
#   trees = make_trees()
#   list_of_languages = get_languages_in_grambank()  
#   states = ['0', '1']
#   number_of_samples = 900
#   number_of_languages = len(list_of_languages)
#   sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
#   number_of_relatedness_bins = 10
#   number_of_distance_bins = 10
#   number_of_simulations = 3
#   number_of_steps = 120
#   substitution_matrix = make_random_substitution_matrix()
#   base_frequencies = make_random_base_frequencies()
#   rate_per_branch_length_per_pair = make_random_rate_per_branch_length_per_pair()
#   base_frequencies_list = [base_frequencies]
#   states_list = [states]
#   borrowability_list = [1.0]
#   substitution_matrix_list = [substitution_matrix]  
#   test_input, test_output, relatedness_array, distance_array = make_all_arrays(trees, list_of_languages, sample, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, number_of_simulations, number_of_relatedness_bins=10, number_of_distance_bins=10) 
#   na_array_1 = np.ones([1, number_of_samples, 1])
#   na_array_2 = np.ones([1, 1, number_of_languages]) 
#   result = search_through_parameters_single_feature(test_input, test_output, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, number_of_relatedness_bins=number_of_relatedness_bins, number_of_distance_bins=number_of_distance_bins, number_of_simulations=number_of_simulations, number_of_steps=number_of_steps)  
#   print(result)
#   truth = {'substitution_matrix': substitution_matrix, 'base_frequencies': base_frequencies, 'rate_per_branch_length_per_pair': rate_per_branch_length_per_pair}
#   print(truth)
# 
# def in_trees(item, trees):
#   for tree in trees:
#     for key in tree:
#       glottocode = find_glottocode(key)      
#       if item == glottocode:
#         return True
#   return False      
# 
# def make_reduced_list_of_languages(list_of_languages, trees):
#   result = []
#   for item in list_of_languages:
#     if in_trees(item, trees):
#       result.append(item)
#   return result
#         
# def search_through_parameters_single_feature_sanity_check():
#   trees = make_trees()
#   list_of_languages = get_languages_in_grambank()  
#   states = ['0', '1']
#   number_of_samples = 900
#   number_of_languages = len(list_of_languages)
#   sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
#   number_of_relatedness_bins = 10
#   number_of_distance_bins = 10
#   number_of_simulations = 3
#   number_of_steps = 60
#   substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
#   base_frequencies = {'0': 1, '1': 0}
#   rate_per_branch_length_per_pair = 0.03
#   base_frequencies_list = [base_frequencies]
#   states_list = [states]
#   borrowability_list = [1.0]
#   substitution_matrix_list = [substitution_matrix]  
#   test_input, test_output, relatedness_array, distance_array = make_all_arrays(trees, list_of_languages, sample, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, number_of_simulations, number_of_relatedness_bins=10, number_of_distance_bins=10) 
#   na_array_1 = np.ones([1, number_of_samples, 1])
#   na_array_2 = np.ones([1, 1, number_of_languages]) 
#   result = search_through_parameters_single_feature(test_input, test_output, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, number_of_relatedness_bins=number_of_relatedness_bins, number_of_distance_bins=number_of_distance_bins, number_of_simulations=number_of_simulations, number_of_steps=number_of_steps)  
#   print(result)
#   truth = {'substitution_matrix': substitution_matrix, 'base_frequencies': base_frequencies, 'rate_per_branch_length_per_pair': rate_per_branch_length_per_pair}
#   print(truth)
# 
# def search_through_parameters_single_feature_sanity_check_reduced():
#   trees = make_trees()
#   list_of_languages = get_languages_in_grambank()
#   remake = False
#   trees = make_reduced_trees(trees, list_of_languages, remake=remake)
#   list_of_languages = make_reduced_list_of_languages(list_of_languages, trees)
#   locations = get_locations(trees, remake=remake)
#   nodes_to_tree_dictionary = make_nodes_to_tree_dictionary(trees, remake=remake)
#   reconstructed_locations_dictionary = make_reconstructed_locations_dictionary(trees, locations, nodes_to_tree_dictionary, remake=remake)
#   time_depths_dictionary = make_time_depths_dictionary(trees, remake=remake)
#   parent_dictionary = make_parent_dictionary(trees, remake=remake)
#   contemporary_neighbour_dictionary = make_contemporary_neighbour_dictionary(trees, reconstructed_locations_dictionary, time_depths_dictionary, parent_dictionary, remake=remake)
#   potential_donors = make_potential_donors(reconstructed_locations_dictionary, time_depths_dictionary, contemporary_neighbour_dictionary, remake=remake)
#   child_dictionary = make_child_dictionary(trees, remake=remake)  
# 
#   context = {}
#   context[TREES] = trees
#   context[LIST_OF_LANGUAGES] = list_of_languages
#   context[LOCATIONS] = locations
#   context[NODES_TO_TREE_DICTIONARY] = nodes_to_tree_dictionary
#   context[RECONSTRUCTED_LOCATIONS_DICTIONARY] = reconstructed_locations_dictionary
#   context[PARENT_DICTIONARY] = parent_dictionary
#   context[CONTEMPORARY_NEIGHBOUR_DICTIONARY] = contemporary_neighbour_dictionary
#   context[POTENTIAL_DONORS] = potential_donors
#   context[CHILD_DICTIONARY] = child_dictionary
#   context[TIME_DEPTHS_DICTIONARY] = time_depths_dictionary
#   
#   states = ['0', '1']
#   number_of_samples = 400
#   number_of_languages = len(list_of_languages)
#   sample = np.random.choice(np.array(list_of_languages), number_of_samples, replace=False)
#   number_of_relatedness_bins = 10
#   number_of_distance_bins = 10
#   number_of_simulations = 20
#   '''
#   maybe increase the number of simulations here
#   '''
#   number_of_steps = 150
#   substitution_matrix = [[0.95, 0.05], [0.05, 0.95]]
#   base_frequencies = {'0': 1, '1': 0}
#   rate_per_branch_length_per_pair = 0.03
#   base_frequencies_list = [base_frequencies]
#   states_list = [states]
#   borrowability_list = [1.0]
#   substitution_matrix_list = [substitution_matrix]  
#   test_input, test_output, relatedness_array, distance_array = make_all_arrays(trees, list_of_languages, sample, substitution_matrix_list, states_list, base_frequencies_list, rate_per_branch_length_per_pair, borrowability_list, number_of_simulations, context, number_of_relatedness_bins=10, number_of_distance_bins=10) 
#   na_array_1 = np.ones([1, number_of_samples, 1])
#   na_array_2 = np.ones([1, 1, number_of_languages]) 
#   result = search_through_parameters_single_feature(test_input, test_output, relatedness_array, distance_array, na_array_1, na_array_2, trees, list_of_languages, sample, states, context, number_of_relatedness_bins=number_of_relatedness_bins, number_of_distance_bins=number_of_distance_bins, number_of_simulations=number_of_simulations, number_of_steps=number_of_steps)  
# #   print(result)
# #   truth = {'substitution_matrix': substitution_matrix, 'base_frequencies': base_frequencies, 'rate_per_branch_length_per_pair': rate_per_branch_length_per_pair}
# #   print(truth)
# 
# 
# 
# 
# '''
# now want to plan a pipeline for testing whether two families had contact or are related
# 
# let's say that you have two families;
# you are taking each language and comparing it to each other language.
# 
# what you could do is just take the classifier that is already trained;
# and all you are changing is the relatedness dictionary.
# 
# you need to have two models trained on new data.
# 
# you can have every language;
# then the model needs to have a weighting for each language, as before.
# 
# i have FamilyTestModel now.
# 
# you need to have the two families you are comparing.
# 
# you have a function for make_all_arrays_relatedness_test(), which makes the input and output array and na_arrays.
# 
# for grambank you similarly need a function make_all_arrays_for_grambank_relatedness_test.
# that function currently takes a value dictionary.  
# 
# pipelines that you need:
# focus first on testing whether two families are related.
# 
# 1. equivalent of make all arrays, where you can specify that two families are related
# 2. pipeline which takes an input and output array and the two families, and tests two hypotheses, one that they are related and one that they are not
# 3. a pipeline for choosing two families and simulating random data for one of those hypotheses, and testing whether pipeline 2 can get it right.
# 
# '''
# 
# 
# '''
# 
# a more efficient pipeline;
# 
# you train the Model on a single simulated dataset.
# you get the weights.
# 
# you then store this.
# 
# you do this with different values for the parameters.
# 
# 
# you then try to make a model which predicts the values of the parameters from the 
# values of the weights.
# 
# one function in this file for simulating data, storing the weights after training the model along 
# with the parameters in two npy files; an input array and output array.
# call them meta_input.npy and meta_output.npy.
# 
# then you need a model class called MetaModel
# 
# what is the structure?
# 
# 
# 
# 
# 
# '''







  
  