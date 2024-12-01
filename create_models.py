import gensim.downloader as api
from gensim.models import KeyedVectors as word2vec
import json, os
from gensim.models import KeyedVectors
import logging
#'VERB_SCORE', 'DET_SCORE', 'ENGLISHV_SCORE', 'POSITION_RATIO','METHODV_SCORE', 'CONTAINSLISTVERB'
stable_features = ['WORD', 'SPLIT_IDENTIFIER', 'CONTEXT_NUMBER'] #'LANGUAGE' 'PREP_SCORE'
mutable_feature_list = ['ENGLISHV_SCORE', 'POSITION', 'MAXPOSITION', 'NORMALIZED_POSITION','POSITION_RATIO', 'VERB_SCORE', 'CONTAINSDIGITS', 'CONTAINSVERB', 'CONTAINSCLOSEDSET', 'CONJ_SCORE'] # 'ENGLISHN_SCORE' ,'ENGLISHPRE_SCORE'  'ENGLISHN_SCORE', 'METHODN_SCORE', 'METHODV_SCORE', 'CODEPRE_SCORE', 'ENGLISHV_SCORE', 'METHODPRE_SCORE', 'ENGLISHPRE_SCORE', 'CONJ_SCORE'

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
    Checks for native KeyedVectors format, converts from txt if needed.
    
    Args:
        pklFile (str, optional): The path to a pickle file. Defaults to an empty string.
        rootDir (str): Root directory containing model files
    
    Returns:
        tuple: A tuple containing three Word2Vec models: (modelGensimTokens, modelGensimMethods, modelGensimEnglish).
    
    Raises:
        FileNotFoundError: If text vector files are missing
        PermissionError: If there are permission issues with file access
        IOError: For other I/O related errors
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        # Load FastText model 
        logger.info("Loading FastText model...")
        modelGensimEnglish = api.load('word2vec-google-news-300')
        logger.info("FastText model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load FastText model: {e}")
        raise
    
    # Paths for txt and native format files
    token_txt_path = os.path.join(rootDir, 'code2vec', 'token_vecs.txt')
    token_native_path = os.path.join(rootDir, 'code2vec', 'token_vecs.kv')
    
    method_txt_path = os.path.join(rootDir, 'code2vec', 'target_vecs.txt')
    method_native_path = os.path.join(rootDir, 'code2vec', 'target_vecs.kv')
    
    # Load token vectors
    try:
        if not os.path.exists(token_txt_path):
            raise FileNotFoundError(f"Token vector text file not found: {token_txt_path}")
        
        if not os.path.exists(token_native_path):
            logger.info("Native token vector format not found. Converting...")
            try:
                modelGensimTokens = KeyedVectors.load_word2vec_format(token_txt_path, binary=False)
                modelGensimTokens.save(token_native_path)
                logger.info(f"Token vectors converted and saved to {token_native_path}")
            except PermissionError:
                logger.error(f"Permission denied when saving token vectors to {token_native_path}")
                raise
            except IOError as e:
                logger.error(f"Error converting token vectors: {e}")
                raise
        else:
            modelGensimTokens = KeyedVectors.load(token_native_path)
            logger.info("Token vectors loaded from native format")
    except Exception as e:
        logger.error(f"Error loading token vectors: {e}")
        raise
    
    # Load method vectors
    try:
        if not os.path.exists(method_txt_path):
            raise FileNotFoundError(f"Method vector text file not found: {method_txt_path}")
        
        if not os.path.exists(method_native_path):
            logger.info("Native method vector format not found. Converting...")
            try:
                modelGensimMethods = KeyedVectors.load_word2vec_format(method_txt_path, binary=False)
                modelGensimMethods.save(method_native_path)
                logger.info(f"Method vectors converted and saved to {method_native_path}")
            except PermissionError:
                logger.error(f"Permission denied when saving method vectors to {method_native_path}")
                raise
            except IOError as e:
                logger.error(f"Error converting method vectors: {e}")
                raise
        else:
            modelGensimMethods = KeyedVectors.load(method_native_path)
            logger.info("Method vectors loaded from native format")
    except Exception as e:
        logger.error(f"Error loading method vectors: {e}")
        raise
    
    logger.info("All models loaded successfully")
    return modelGensimTokens, modelGensimMethods, modelGensimEnglish