import json, os
import logging
#'VERB_SCORE', 'DET_SCORE', 'ENGLISHV_SCORE', 'POSITION_RATIO','METHODV_SCORE', 'CONTAINSLISTVERB'
stable_features = ['WORD', 'SPLIT_IDENTIFIER', 'CONTEXT_NUMBER'] #'LANGUAGE' 'PREP_SCORE' 'CONTAINSLISTVERB','CONTAINSCLOSEDSET'
mutable_feature_list = ['ENGLISHPRE_SCORE', 'PREPOSITION', 'DETERMINER', 'PREP_SCORE', 'NORMALIZED_POSITION', 'DET_SCORE', 'CONSONANT_VOWEL_RATIO', 'PREV_POS', 'NEXT_POS', 'NLTK_POS', 'MORPHOLOGICAL_PLURAL', 'POSITION', 'MAXPOSITION','POSITION_RATIO', 'ENGLISH_VERB_SCORE', 'CONJ_SCORE','ENGLISH_NOUN_SCORE'] 
columns_to_drop = ['SPLIT_IDENTIFIER', 'WORD', 'MAXPOSITION', 'POSITION']

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
    Handles missing files gracefully by setting models to None instead of raising errors.
    
    Args:
        pklFile (str, optional): The path to a pickle file. Defaults to an empty string.
        rootDir (str): Root directory containing model files
    
    Returns:
        tuple: A tuple containing three Word2Vec models: 
               (modelGensimTokens, modelGensimMethods, modelGensimEnglish).
               Models that fail to load are set to None.
    """
    import gensim.downloader as api
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    modelGensimEnglish = None
    modelGensimTokens = None
    modelGensimMethods = None

    # Attempt to load FastText model
    try:
        logger.info("Attempting to load local FastText model...")
        # The model should be in the gensim-data directory after download
        model_path = os.path.expanduser('~/gensim-data/fasttext-wiki-news-subwords-300/fasttext-wiki-news-subwords-300.model')
        
        if os.path.exists(model_path):
            import gensim
            modelGensimEnglish = gensim.models.fasttext.load_facebook_model(model_path)
            logger.info("Local FastText model loaded successfully")
        else:
            logger.info("Local model not found, attempting to download...")
            modelGensimEnglish = api.load('fasttext-wiki-news-subwords-300')
            logger.info("FastText model downloaded and loaded successfully")
    except Exception as e:
        logger.warning(f"FastText model could not be loaded: {e}")

    # Paths for token vectors
    token_txt_path = os.path.join(rootDir, 'code2vec', 'token_vecs.txt')
    token_native_path = os.path.join(rootDir, 'code2vec', 'token_vecs.kv')
    
    # Paths for method vectors
    method_txt_path = os.path.join(rootDir, 'code2vec', 'target_vecs.txt')
    method_native_path = os.path.join(rootDir, 'code2vec', 'target_vecs.kv')

    return modelGensimTokens, modelGensimMethods, modelGensimEnglish