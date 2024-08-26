import os
import time
import joblib
import nltk
import pandas as pd
from feature_generator import *
from flask import Flask
from waitress import serve
from spiral import ronin
import json

app = Flask(__name__)

class ModelData:
    def __init__(self, ModelTokens, ModelMethods, ModelGensimEnglish) -> None:
        """
        Initialize an instance of the ModelData class with word vector models.

        Args:
            ModelTokens: Word vectors model for tokens.
            ModelMethods: Word vectors model for methods.
            ModelGensimEnglish: Word vectors model for general English words.
        """
        self.ModelTokens = ModelTokens
        self.ModelMethods = ModelMethods
        self.ModelGensimEnglish = ModelGensimEnglish
        self.ModelClassifier = joblib.load('output/model_RandomForestClassifier.pkl')

class AppCache:
    def __init__(self, Path, Filename) -> None:
        self.Cache = {}
        self.Path = Path
        self.Filename = Filename

    def load(self): 
        if not os.path.isdir(self.Path): 
            raise Exception("Cannot load path: "+self.Path)
        else:
            if not os.path.isfile(self.Path+"/"+self.Filename):
                JSONcache = open(self.Path+"/"+self.Filename, 'w')
                json.dump({}, JSONcache)
                JSONcache.close()
            JSONcache = open(self.Path+"/"+self.Filename, 'r')
            self.Cache = json.load(JSONcache)
            JSONcache.close()

    def add(self, identifier, result):
        info = result
        info.update({"firstEncounter": time.time()})
        info.update({"lastEncounter": time.time()})
        info.update({"count": 1})
        info.update({"version": "SCANL 1.0"})
        self.Cache.update({identifier : info})

    def encounter(self, identifier):
        self.Cache[identifier].update({"lastEncounter": time.time()})
        self.Cache[identifier].update({"count": self.Cache[identifier]["count"]+1})
        self.Cache[identifier].update({"version": "SCANL 1.0"})

    def save(self):
        JSONcache = open(self.Path+"/"+self.Filename, 'w')
        json.dump(self.Cache, JSONcache)
        JSONcache.close()

def initialize_model():
    """
    Initialize and load word vectors for the application.

    This function initializes and loads word vectors using the 'createModel' function and stores them in the
    'app.model_data' attribute for later use.

    Returns:
        None
    """
    print("Loading word vectors!!")
    ModelTokens, ModelMethods, ModelGensimEnglish = createModel()
    print("Word vectors loaded!!")

    app.model_data = ModelData(ModelTokens, ModelMethods, ModelGensimEnglish)

def start_server(temp_config = {}):
    """
    Initialize the model and start the server.

    This function first initializes the model by calling the 'initialize_model' function. Then, it starts the server using
    the waitress `serve` method, allowing incoming HTTP requests to be handled.

    The arguments to waitress serve are read from the configuration file `serve.json`. The default option is to
    listen for HTTP requests on all interfaces (ip address 0.0.0.0, port 5000).

    Returns:
        None
    """
    print('initializing model...')
    initialize_model()

    print("loading cache...")
    if not os.path.isdir("cache"): os.mkdir("cache")
    app.cache = AppCache("cache", "cache.json")
    app.studentCache = AppCache("cache", "student_cache.json")
    app.cache.load()

    print("loading dictionary...")
    nltk.download("words")
    app.english_words = set(w.lower() for w in nltk.corpus.words.words())

    print('retrieving server configuration...')
    data = open('serve.json')
    config = json.load(data)

    server_host = temp_config["address"] if "address" in temp_config.keys() else config["address"]
    server_port = temp_config["port"] if "port" in temp_config.keys() else config['port']
    server_url_scheme = temp_config["protocol"] if "protocol" in temp_config.keys() else config["protocol"]

    print("Starting server...")
    serve(app, host=server_host, port=server_port, url_scheme=server_url_scheme)
    data.close()

def dictionary_lookup(word):
    #return true if the word exists in the dictionary (the nltk words corpus)
    return word.lower() in app.english_words

#TODO: add functions here to look for abbreviations and slang

#TODO: this is not an intuitive way to save cache
@app.route('/')
def save():
    app.cache.save()
    app.studentCache.save()
    return "successfully saved cache"

@app.route('/<student>/<identifier_name>/<identifier_context>')
def listen(student, identifier_name, identifier_context):
    #check if identifier name has already been used
    cache = None;

    if (student == "student"):
        cache = app.studentCache
    else: 
        cache = app.cache

    if (identifier_name in cache.Cache.keys()): 
        cache.encounter(identifier_name)
        return cache.Cache[identifier_name]

    """
    Process a web request to analyze an identifier within a specific context.

    This route function takes two URL parameters (identifier_name, and identifier_context) from an
    incoming HTTP request and performs several data preprocessing and feature extraction steps on the identifier_name.
    It then uses a trained classifier to annotate the identifier with part-of-speech tags and other linguistic features.

    Args:
        identifier_name (str): The name of the identifier to be analyzed.
        identifier_context (str): The context in which the identifier appears.

    Returns:
        list: A list of annotations for the identifier, including part-of-speech tags and other linguistic features.
    """
    print("INPUT: {ident_name} {ident_context}".format(ident_name=identifier_name, ident_context=identifier_context))
    
    words = ronin.split(identifier_name)
    max_pos = [len(words) for word in words]
    position = [pos for pos in range(len(words))]
    context_type = [identifier_context for _ in range(len(words))]
    normalized_pos = [0 if i == 0 else (2 if i == len(words) - 1 else 1) for i in range(len(words))]
    
    word_data = pd.DataFrame(words, columns=['WORD'])
    pos_data = pd.DataFrame(max_pos, columns=['MAXPOSITION'])
    normalized_data = pd.DataFrame(normalized_pos, columns=['NORMALIZED_POSITION'])
    position_data = pd.DataFrame(position, columns=['POSITION'])
    context_data = pd.DataFrame(context_type, columns=['CONTEXT'])
    
    context_data['CONTEXT'] = context_data['CONTEXT'].apply(context_to_number)
    data = pd.concat([word_data, pos_data, normalized_data, context_data, position_data], axis=1)
    
    data = createVerbVectorFeature(data, app.model_data.ModelGensimEnglish)
    data = createDeterminerVectorFeature(data, app.model_data.ModelGensimEnglish)
    data = createConjunctionVectorFeature(data, app.model_data.ModelGensimEnglish)
    data = createPrepositionVectorFeature(data, app.model_data.ModelGensimEnglish)
    data = createPreambleVectorFeature("CODE", data, app.model_data.ModelTokens)
    data = createPreambleVectorFeature("METHOD", data, app.model_data.ModelMethods)
    data = createPreambleVectorFeature("ENGLISH", data, app.model_data.ModelGensimEnglish)
    data = createLetterFeature(data)
    data = wordPosTag(data)
    
    data['NLTK_POS'] = data['NLTK_POS'].astype('category')
    data['NLTK_POS'] = data['NLTK_POS'].cat.codes

    data = createSimilarityToVerbFeature("METHODV", app.model_data.ModelMethods, data)
    data = createSimilarityToVerbFeature("ENGLISHV", app.model_data.ModelGensimEnglish, data)
    data = createSimilarityToNounFeature("METHODN", app.model_data.ModelMethods, data)
    data = createSimilarityToNounFeature("ENGLISHN", app.model_data.ModelGensimEnglish, data)
    data = createDeterminerFeature(data)
    data = createDigitFeature(data)
    data = createPrepositionFeature(data)

    # input_model = 'output/model_RandomForestClassifier.pkl'
    # clf = joblib.load(input_model)

    # result = {
    #     "tags" : list(annotate_identifier(app.model_data.ModelClassifier, data)),
    #     "words" : words
    # }

    # create response JSON
    tags = list(annotate_identifier(app.model_data.ModelClassifier, data))
    result = {
        "words" : []
    }

    for i in range(len(words)):
        #check dictionary
        dictionary = "UC" #uncategorized
        if (dictionary_lookup(words[i])): dictionary = "DW" #dictionary word
        #append result
        result["words"].append(
            {
                words[i] : {
                    "tag" : tags[i],
                    "dictionary" : dictionary
                }
            }
        )

    # append result to cache
    cache.add(identifier_name, result)

    return result
    
def context_to_number(context):
    """
    Convert a textual context description to a numerical representation.

    This function takes a context description as a string and maps it to a numerical representation according to a
    predefined mapping.

    Args:
        context (str): The textual context description.

    Returns:
        int: The numerical representation of the context.

    Raises:
        ValueError: If the provided context is not one of the predefined values.

    Example:
        numeric_context = context_to_number("CLASS")
    """
    if context == "ATTRIBUTE":
        return 1
    elif context == "CLASS":
        return 2
    elif context == "DECLARATION":
        return 3
    elif context == "FUNCTION":
        return 4 
    elif context == "PARAMETER":
        return 5

def annotate_identifier(clf, data):
    """
    Annotate identifier tokens using a trained classifier.

    This function takes a trained classifier and a dataset containing features for identifier tokens. It applies the
    classifier to predict labels for the identifier tokens.

    Args:
        clf (Classifier): The trained classifier model.
        data (pd.DataFrame): A DataFrame containing features for identifier tokens. The columns of the DataFrame should
                             match the feature names used during training.

    Returns:
        np.array: An array of predicted labels for the identifier tokens.

    Example:
        clf = load_classifier('trained_classifier.pkl')
        data = pd.DataFrame(...)  # Create a DataFrame with feature data.
        predictions = annotate_identifier(clf, data)
    """
    
    independent_variables_add = ['NORMALIZED_POSITION', 'LAST_LETTER', 'CONTEXT', 'MAXPOSITION', 'NLTK_POS', 'POSITION',
                                 'VERB_SCORE', 'DET_SCORE', 'PREP_SCORE', 'CONJ_SCORE', 'PREPOSITION', 'DETERMINER',
                                 'ENGLISHV_SCORE', 'ENGLISHN_SCORE', 'METHODN_SCORE', 'METHODV_SCORE', 'CODEPRE_SCORE',
                                 'METHODPRE_SCORE', 'ENGLISHPRE_SCORE']
    
    df_features = pd.DataFrame(data, columns=independent_variables_add)
    y_pred = clf.predict(df_features)
    return y_pred
