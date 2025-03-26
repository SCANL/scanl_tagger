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
import sqlite3
from create_models import createModel, mutable_feature_list

app = Flask(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
class ModelData:
    def __init__(self, modelTokens, modelMethods, modelGensimEnglish, wordCount) -> None:
        """
        Initialize an instance of the ModelData class with word vector models.

        Args:
            ModelTokens: Word vectors model for tokens.
            ModelMethods: Word vectors model for methods.
            ModelGensimEnglish: Word vectors model for general English words.
        """

        self.ModelTokens = modelTokens
        self.ModelMethods = modelMethods
        self.ModelGensimEnglish = modelGensimEnglish
        self.wordCount = wordCount

class CacheIndex:
    def __init__(self, Path) -> None:
        self.Path = Path
        #create a table that just has a single column of cache IDs
        conn = sqlite3.connect(Path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS caches (
                cache_id TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def add(self, cache_id):
        #add cache_id to the table
        conn = sqlite3.connect(self.Path)
        cursor = conn.cursor()
        #cache_id needs to be by itself in a tuple for some reason? otherwise sqlite freaks out idk
        cursor.execute('''
            INSERT INTO caches (cache_id) VALUES (?)
        ''', (cache_id,))
        conn.commit()
        conn.close()

    def isCacheExistent(self, cache_id):
        conn = sqlite3.connect(self.Path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT cache_id FROM caches WHERE cache_id = ?
        ''', (cache_id,))
        row = cursor.fetchone()
        if row: return True
        else: return False


#TODO: context should probably be considered when saving tagged names
class AppCache:
    def __init__(self, Path) -> None:
        self.Path = Path #path to an SQL lite database
    
    def load(self):
        #create connection to database
        conn = sqlite3.connect(self.Path)
        #create the table of names if it doesn't exist
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS names (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       name TEXT NOT NULL,
                       words TEXT, -- this is a JSON string
                       firstEncounter INTEGER,
                       lastEncounter INTEGER,
                       count INTEGER
                       )
        ''')
        #close the database connection
        conn.commit()
        conn.close()

    def add(self, identifier, result):
        #connection setup
        conn = sqlite3.connect(self.Path)
        cursor = conn.cursor()
        #add identifier to table
        record = {
            "name": identifier,
            "words": json.dumps(result["words"]),
            "firstEncounter": time.time(),
            "lastEncounter": time.time(),
            "count": 1
        }
        cursor.execute('''
            INSERT INTO names (name, words, firstEncounter, lastEncounter, count)
            VALUES (:name, :words, :firstEncounter, :lastEncounter, :count)
        ''', record)
        #close the database connection
        conn.commit()
        conn.close()

    def retrieve(self, identifier):
        #return a dictionary of the name, or false if not in database
        conn = sqlite3.connect(self.Path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, words, firstEncounter, lastEncounter, count FROM names WHERE name = ?", (identifier,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "name": row[0],
                "words": json.loads(row[1]),
                "firstEncounter": row[2],
                "lastEncounter": row[3],
                "count": row[4]
            }
        else: 
            return False

    def encounter(self, identifier):
        currentCount = self.retrieve(identifier)["count"]
        #connection setup
        conn = sqlite3.connect(self.Path)
        cursor = conn.cursor()
        #update record
        cursor.execute('''
            UPDATE names 
            SET lastEncounter = ?, count = ?
            WHERE name = ?
        ''', (time.time(), currentCount+1, identifier))
        #close connection
        conn.commit()
        conn.close()

class WordList:
    def __init__(self, Path):
        self.Words = set()
        self.Path = Path
    
    def load(self):
        if not os.path.isfile(self.Path):
            print("Could not find word list file!")
            return
        with open(self.Path) as file: 
            for line in file:
                self.Words.add(line[:line.find(',')]) #stop at comma
    
    def find(self, item):
        return item in self.Words

def initialize_model():
    """
    Initialize and load word vectors for the application, and load a word count DataFrame.

    This function initializes and loads word vectors using the 'createModel' function, and loads word counts 
    from a JSON file into a Pandas DataFrame for use in the application.

    Returns:
        tuple: (ModelData, WORD_COUNT DataFrame)
    """
    print("Loading word vectors!!")
    modelTokens, modelMethods, modelGensimEnglish = createModel(rootDir=SCRIPT_DIR)
    print("Word vectors loaded!!")
    
    # Load the word count JSON file into a DataFrame
    word_count_path = os.path.join("input", "word_count.json")
    if os.path.exists(word_count_path):
        print(f"Loading word count data from {word_count_path}...")
        word_count_df = pd.read_json(word_count_path, orient='index', typ='series').reset_index()
        word_count_df.columns = ['word', 'log_frequency']
        print("Word count data loaded!")
    else:
        print(f"Word count file not found at {word_count_path}. Initializing empty DataFrame.")
        word_count_df = pd.DataFrame(columns=['word', 'log_frequency'])
    
    # Create and store model data
    app.model_data = ModelData(modelTokens, modelMethods, modelGensimEnglish, word_count_df)

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

    print("setting up cache...")
    if not os.path.exists('cache'): os.mkdir('cache')
    app.cacheIndex = CacheIndex('cache/index.db')

    print("loading dictionary...")
    nltk.download("words")
    app.english_words = set(w.lower() for w in nltk.corpus.words.words())
    #insert english words from words/en.txt
    if not os.path.exists("words/en.txt"):
        print("could not find English words, using WordNet only!")
    else:
        with open("words/en.txt") as words:
            for word in words:
                app.english_words.add(word[:-1])

    print('retrieving server configuration...')
    data = open('serve.json')
    config = json.load(data)

    server_host = temp_config["address"] if "address" in temp_config.keys() else config["address"]
    server_port = temp_config["port"] if "port" in temp_config.keys() else config['port']
    server_url_scheme = temp_config["protocol"] if "protocol" in temp_config.keys() else config["protocol"]

    print("loading word list...")
    wordListPath = temp_config["words"] if "words" in temp_config.keys() else config["words"]
    app.words = WordList(wordListPath)
    app.words.load()

    print("Starting server...")
    serve(app, host=server_host, port=server_port, url_scheme=server_url_scheme)
    data.close()

def dictionary_lookup(word):
    #return true if the word exists in the dictionary (the nltk words corpus)
    #or if the word is in the list of approved words
    dictionaryType = ""
    dictionary = word.lower() in app.english_words
    acceptable = app.words.find(word)
    digit = word.isnumeric()
    if (dictionary):
        dictionaryType = "DW"
    elif (acceptable):
        dictionaryType = "AW"
    elif (digit):
        dictionaryType = "DD"
    else:
        dictionaryType = "UC"
    
    return dictionaryType

#caches should be saved in an SQL lite database
@app.route('/<identifier_name>/<identifier_context>')
@app.route('/<identifier_name>/<identifier_context>/<cache_id>')
def listen(identifier_name: str, identifier_context: str, cache_id: str = None) -> List[dict]:
    #check if identifier name has already been used
    cache = None
    #find the existing cache in app.caches or create a new one if it doesn't exist
    if cache_id != None:
        if app.cacheIndex.isCacheExistent(cache_id):
            #check if the identifier name is in this cache and return it if so
            cache = AppCache("cache/"+cache_id+".db")
            cache.encounter(identifier_name)
            data = cache.retrieve(identifier_name)
            if data != False:
                return data
        else:
            #create the cache and add it to the dictionary of caches
            cache = AppCache("cache/"+cache_id+".db")
            cache.load()
            app.cacheIndex.add(cache_id)
    
    """
    Process a web request to analyze an identifier within a specific context.

    This route function takes two URL parameters (identifier_name, and identifier_context) from an
    incoming HTTP request and performs data preprocessing and feature extraction on the identifier_name.
    It then uses a trained classifier to annotate the identifier with part-of-speech tags and other linguistic features.

    Args:
        identifier_name (str): The name of the identifier to be analyzed.
        identifier_context (str): The context in which the identifier appears.

    Returns:
        List[dict]: A list of dictionaries containing words and their predicted POS tags.
    """
    print(f"INPUT: {identifier_name} {identifier_context}")
   
    # Split identifier_name into words
    words = ronin.split(identifier_name)

    # # Create initial data frame
    data = pd.DataFrame({
        'WORD': words,
        'SPLIT_IDENTIFIER': ' '.join(words),
        'CONTEXT_NUMBER': context_to_number(identifier_context),  # Predefined context number
    })

    # create response JSON
    # tags = list(annotate_identifier(app.model_data.ModelClassifier, data))
    result = {
        "words" : []
    }

    # Add features to the data
    data = createFeatures(
        data, 
        mutable_feature_list,
        modelGensimEnglish=app.model_data.ModelGensimEnglish,
    )
    
    categorical_features = ['NLTK_POS']
    category_variables = []

    for category_column in categorical_features:
        if category_column in data.columns:
            category_variables.append(category_column)
            data.loc[:, category_column] = data[category_column].astype(str)

    for category_column in category_variables:
        # Explicitly handle categorical conversion
        unique_values = data[category_column].unique()
        category_map = {}
        for value in unique_values:
            if value in universal_to_custom:
                category_map[value] = custom_to_numeric[universal_to_custom[value]]
            else:
                category_map[value] = custom_to_numeric['NOUN']  # Assign 'NM' (8) for unknown categories

        data.loc[:, category_column] = data[category_column].map(category_map)

    # Convert categorical variables to numeric
    # Load and apply the classifier
    clf = joblib.load(os.path.join(SCRIPT_DIR, 'output', 'model_GradientBoostingClassifier.pkl'))
    predicted_tags = annotate_identifier(clf, data)

    # Combine words and their POS tags into a parseable format
    #result = [{'word': word, 'pos_tag': tag} for word, tag in zip(words, predicted_tags)]

    for i in range(len(words)):
        #check dictionary
        dictionary = "UC" #uncategorized
        word = words[i]
        dictionary = dictionary_lookup(word)
        result["words"].append(
            {
                words[i] : {
                    "tag" : predicted_tags[i],
                    "dictionary" : dictionary
                }
            }
        )

    # append result to cache
    if cache_id != None:
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
    """
    # Drop unnecessary columns
    data = data.drop(columns=['WORD', 'SPLIT_IDENTIFIER'], errors='ignore')

    # Ensure only the features used during training are included
    trained_features = clf.feature_names_in_  # Features expected by the classifier
    missing_features = set(trained_features) - set(data.columns)
    extra_features = set(data.columns) - set(trained_features)

    if missing_features:
        raise ValueError(f"The following expected features are missing: {missing_features}")
    if extra_features:
        print(f"Warning: The following unused features are being ignored: {extra_features}")
        data = data[trained_features]

    # Ensure feature order matches the trained model
    df_features = data[trained_features]
    
    print("THESE")
    print(df_features)
    
    print("THOSE")
    print(clf.feature_names_in_)

    # Make predictions
    y_pred = clf.predict(df_features)
    return y_pred
