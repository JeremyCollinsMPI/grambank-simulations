from CreateDataFrame import *
from TreeFunctions import *
from copy import deepcopy

def get_languages_in_grambank():
  data = readData('data.txt')
  # dict = createDictionary(df)
  languages = getUniqueLanguages(data)
  return languages





def only_retain_included_languages(trees, list_of_languages):
#   for i in range(len(trees)):
  for i in range(10):

    print(i)
    tree = trees[i]
    tree = tree.strip('\n')
    new_tree = createTree(tree)
    tips = findTips(new_tree)
    for tip in tips:
      glottocode = find_glottocode(findNodeName(tip))
      if not glottocode in list_of_languages:
        print(new_tree)
        print(tip)
        try:
          new_tree = dropNode(new_tree, tip)
        except:
          pass
    trees[i] = new_tree
    print(new_tree)
  return trees
    
    
    
    
#     keys = deepcopy(list(new_tree.keys()))
# #     print(keys)
#     for key in keys:
#       print(findNodeName(key))
#       if not findNodeName(key) in list_of_languages:
#         try:
#           new_tree = dropNode(new_tree, key)
# #           print('managed to drop ', key)
#         except:
#           print('problem: ', key)
#           pass
#     trees[i] = new_tree
#     print(new_tree)
#   return trees