import warnings
from asl_data import SinglesData


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

    # iterate through all the test word sequences
    for word_sequence, (index, sequence) in test_set.get_all_Xlengths().items():
        # initialize values
        best_score = float("-inf")
        score = float("-inf")
        best_guess = ""
        probability_dic = {}

        # iterate though all the HMM models and calculate the score between the model and the test word sequence
        for word, model in models.items():
            # try/ except construct as necessary to eliminate non-viable models
            try:
                score = model.score(index, sequence)
                probability_dic[word] = score
            except:
                probability_dic[word] = float("-inf")

            # if better score save the word as best guess so far
            if score > best_score:
                best_score = score
                best_guess = word

        probabilities.append(probability_dic)
        guesses.append(best_guess)

    return probabilities, guesses
