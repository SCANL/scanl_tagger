import gensim.downloader as api
from gensim.models import KeyedVectors as word2vec
import json, os
from gensim.models import KeyedVectors
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
    
    # Helper function to load models safely
    def load_model(txt_path, native_path, model_name):
        """
        Load a word vector model, converting from text format if necessary.
        
        Args:
            txt_path (str): Path to the text-based word vectors.
            native_path (str): Path to the native .kv format file.
            model_name (str): Name of the model for logging.

        Returns:
            KeyedVectors or None: The loaded model, or None if unavailable.
        """
        try:
            if os.path.exists(native_path):
                logger.info(f"Loading {model_name} from native format...")
                return KeyedVectors.load(native_path)
            
            elif os.path.exists(txt_path):
                logger.info(f"Native format for {model_name} not found. Converting from text format...")
                model = KeyedVectors.load_word2vec_format(txt_path, binary=False)
                try:
                    model.save(native_path)
                    logger.info(f"{model_name} vectors converted and saved to {native_path}")
                except PermissionError:
                    logger.warning(f"Permission denied when saving {model_name} to {native_path}. Using in-memory only.")
                return model
            
            else:
                logger.warning(f"{model_name} vector file not found at {txt_path} or {native_path}. Skipping.")
                return None
        
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            return None

    # Load models with the new safe function
    modelGensimTokens = load_model(token_txt_path, token_native_path, "Token vectors")
    modelGensimMethods = load_model(method_txt_path, method_native_path, "Method vectors")

    logger.info("Model loading complete.")
    return modelGensimTokens, modelGensimMethods, modelGensimEnglish