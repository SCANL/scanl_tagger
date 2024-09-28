import gensim.downloader as api
from gensim.models import KeyedVectors as word2vec
import json, os

stable_features = ['NORMALIZED_POSITION', 'MAXPOSITION', 'CONTEXT_NUMBER', 'POSITION', 'LANGUAGE']
mutable_feature_list = ['LAST_LETTER', 'NLTK_POS', 'VERB_SCORE', 'DET_SCORE', 'PREP_SCORE',
                        'CONJ_SCORE', 'PREPOSITION', 'DETERMINER', 'ENGLISHV_SCORE', 'CONTAINSLISTVERB',
                        'ENGLISHN_SCORE', 'METHODN_SCORE', 'METHODV_SCORE', 'CODEPRE_SCORE', 'WORD_COUNT',
                        'METHODPRE_SCORE', 'ENGLISHPRE_SCORE', 'CONTAINSDIGIT', 'CONTAINSCLOSEDSET', 'SECOND_LAST_LETTER']

def load_word_count(input_file):
    """
    Loads the word_count dictionary from a JSON file.

    Parameters:
        input_file (str): The path to the input JSON file.

    Returns:
        dict: The loaded word_count dictionary.
    """
    try:
        with open(input_file, 'r') as f:
            word_count = json.load(f)
        print(f"Word counts successfully loaded from {input_file}")
        return word_count
    except Exception as e:
        print(f"Failed to load word counts: {e}")
        return {}
    
def createModel(pklFile="", rootDir=""):
    """
    Create and load Word2Vec models for tokens, methods, and English text.

    This function loads pre-trained Word2Vec models for tokens, methods, and English text. The models are used for various
    natural language processing tasks.

    Args:
        pklFile (str, optional): The path to a pickle file. Defaults to an empty string.

    Returns:
        tuple: A tuple containing three Word2Vec models: (modelGensimTokens, modelGensimMethods, modelGensimEnglish).
    """
    modelGensimEnglish = api.load('fasttext-wiki-news-subwords-300')
    modelGensimTokens = word2vec.load_word2vec_format(os.path.join(rootDir, 'code2vec', 'token_vecs.txt'), binary=False)
    modelGensimMethods = word2vec.load_word2vec_format(os.path.join(rootDir, 'code2vec', 'target_vecs.txt'), binary=False)
    wordCount = load_word_count(os.path.join(rootDir, 'input', 'word_count.json'))
    
    return wordCount, modelGensimTokens, modelGensimMethods, modelGensimEnglish