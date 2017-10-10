import warnings
from asl_data import SinglesData
from numpy import exp

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    test_cases = test_set.get_all_Xlengths()
    for test_id in test_cases.keys():
        test_case = test_cases[test_id]
        test_probabilities = {}
        
        # Iterate Through Words
        best_LogL = float('-Inf')
        best_word = []
        for word, model in models.items():
            if model:
                try:
                    test_probabilities[word] = model.score(test_case[0], test_case[1]) 
                except ValueError:
                    test_probabilities[word] = float('-Inf')    
                if test_probabilities[word] > best_LogL:
                    best_LogL = test_probabilities[word]
                    best_word = word
            else:
                test_probabilities[word] = float('-Inf') 
        # Write to results
        probabilities.append(test_probabilities)
        guesses.append(best_word)
    return probabilities, guesses
