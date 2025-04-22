import numpy as np  # Import numpy for numerical operations
import nltk  # Import nltk for natural language processing tools
# nltk.download('punkt')  # Uncomment this line if you need to download the 'punkt' tokenizer from NLTK

from nltk.stem.porter import PorterStemmer  # Import the PorterStemmer class for word stemming
stemmer = PorterStemmer()  # Create an instance of the PorterStemmer

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)  # Tokenize the sentence into words, punctuation marks, or numbers


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())  # Convert the word to lowercase and apply stemming to get the root form


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]  # Stem each word in the tokenized sentence
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)  # Initialize a numpy array with zeros, with the same length as the vocabulary list
    for idx, w in enumerate(words):  # Loop through the vocabulary list
        if w in sentence_words:  # If the word exists in the sentence
            bag[idx] = 1  # Set the corresponding position in the bag to 1

    return bag  # Return the bag of words array
