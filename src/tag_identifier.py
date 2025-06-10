import os
import time
import joblib
import nltk
import pandas as pd
from flask import Flask, request
from waitress import serve
from spiral import ronin
import json
import sqlite3
from src.tree_based_tagger.feature_generator import createFeatures, universal_to_custom, custom_to_numeric
from src.tree_based_tagger.create_models import createModel, stable_features, mutable_feature_list
from src.lm_based_tagger.distilbert_tagger import DistilBertTagger

app = Flask(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model_type = None
lm_model = None

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

class AppCache:
    def __init__(self, Path) -> None:
        self.Path = Path

    def load(self):
        #create connection to database
        conn = sqlite3.connect(self.Path)
        #create the table of names if it doesn't exist
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS names (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       name TEXT NOT NULL,
                       context TEXT NOT NULL,
                       words TEXT, -- this is a JSON string
                       firstEncounter INTEGER,
                       lastEncounter INTEGER,
                       count INTEGER,
                       tagTime INTEGER -- time it took to tag the identifier
                       )
        ''')
        #close the database connection
        conn.commit()
        conn.close()

    def add(self, identifier, result, context, tag_time):
        #connection setup
        conn = sqlite3.connect(self.Path)
        cursor = conn.cursor()
        #add identifier to table
        record = {
            "name": identifier,
            "context": context,
            "words": json.dumps(result["words"]),
            "firstEncounter": time.time(),
            "lastEncounter": time.time(),
            "count": 1,
            "tagTime": tag_time
        }
        cursor.execute('''
            INSERT INTO names (name, context, words, firstEncounter, lastEncounter, count, tagTime)
            VALUES (:name, :context, :words, :firstEncounter, :lastEncounter, :count, :tagTime)
        ''', record)
        #close the database connection
        conn.commit()
        conn.close()
        
    def retrieve(self, identifier, context):
        #return a dictionary of the name, or false if not in database
        conn = sqlite3.connect(self.Path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, words, firstEncounter, lastEncounter, count FROM names WHERE name = ? AND context = ?", (identifier, context))
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

    def encounter(self, identifier, context):
        currentCount = self.retrieve(identifier, context)["count"]
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

def initialize_model(temp_config = {}):
    """
    Initialize and load word vectors for the application, and load a word count DataFrame.

    This function initializes and loads word vectors using the 'createModel' function, and loads word counts 
    from a JSON file into a Pandas DataFrame for use in the application.

    Returns:
        tuple: (ModelData, WORD_COUNT DataFrame)
    """
    global model_type, lm_model
    model_type = temp_config.get("model_type", "tree_based")
    if model_type == "tree_based":
        print("Loading word vectors!!")
        modelTokens, modelMethods, modelGensimEnglish = createModel(rootDir=SCRIPT_DIR)
        print("Word vectors loaded!!")
        word_count_path = os.path.join("input", "word_count.json")
        if os.path.exists(word_count_path):
            print(f"Loading word count data from {word_count_path}...")
            word_count_df = pd.read_json(word_count_path, orient='index', typ='series').reset_index()
            word_count_df.columns = ['word', 'log_frequency']
        else:
            print(f"Word count file not found at {word_count_path}. Initializing empty DataFrame.")
            word_count_df = pd.DataFrame(columns=['word', 'log_frequency'])
        app.model_data = ModelData(modelTokens, modelMethods, modelGensimEnglish, word_count_df)
    elif model_type == "lm_based":
        print("Loading DistilBERT tagger...")
        is_local = temp_config.get("local", False)
        lm_model = DistilBertTagger(temp_config['model'], local=is_local)
        print("DistilBERT tagger loaded!")

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
    selected_model = temp_config.get("model_type", "tree_based")
    initialize_model(temp_config)

    print("loading cache...")
    if not os.path.isdir("cache"): os.mkdir("cache")

    print("loading dictionary")
    app.english_words = set(w.lower() for w in nltk.corpus.words.words())

    #insert english words from words/en.txt
    if not os.path.exists("words/en.txt"):
        print("could not find English words, using WordNet only!")
    else:
        with open("words/en.txt") as words:
            for word in words:
                app.english_words.add(word[:-1])

    print('retrieving server configuration...')
    data = open(os.path.join(SCRIPT_DIR, '..', 'serve.json'))
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

#route to check for and create a database if it does not exist already
@app.route('/probe/<cache_id>')
def probe(cache_id: str):
    if os.path.exists("cache/"+cache_id+".db3"):
        return "Opening existing identifier database..."
    else:
        return "First request will create identifier database: "+cache_id+"..."

#route to tag an identifier name
@app.route('/<identifier_name>/<identifier_context>')
@app.route('/<identifier_name>/<identifier_context>/<cache_id>')
def listen(identifier_name: str, identifier_context: str, cache_id: str = None) -> list[dict]:
    # --- Cache lookup (unchanged) ---
    cache = None
    if cache_id is not None:
        if os.path.exists("cache/" + cache_id + ".db3"):
            cache = AppCache("cache/" + cache_id + ".db3")
            data = cache.retrieve(identifier_name, identifier_context)
            if data is not False:
                cache.encounter(identifier_name, identifier_context)
                return data
        else:
            cache = AppCache("cache/" + cache_id + ".db3")
            cache.load()

    # Pull query‐string parameters
    system_name = request.args.get("system_name", default="")
    programming_language = request.args.get("language", default="")
    data_type = request.args.get("type", default="")

    print(f"INPUT: {identifier_name} {identifier_context}")
    start_time = time.perf_counter()

    # 1) Split the identifier into tokens for **both** modes
    words = ronin.split(identifier_name)

    # 2) If we asked for the LM‐based (DistilBERT) tagger, use it
    if model_type == "lm_based":
        result = { "words": [] }

        tags = lm_model.tag_identifier(
            tokens=words,
            context=identifier_context,
            type_str=data_type,
            language=programming_language,
            system_name=system_name
        )

        for word, tag in zip(words, tags):
            dictionary = dictionary_lookup(word)
            result["words"].append({
                word: { "tag": tag, "dictionary": dictionary }
            })

        tag_time = time.perf_counter() - start_time
        if cache_id:
            AppCache(f"cache/{cache_id}.db3").add(identifier_name, result, identifier_context, tag_time)
        return result

    # 3) Else: use the existing tree‐based tagger
    # Create initial DataFrame
    data = pd.DataFrame({
        'WORD': words,
        'SPLIT_IDENTIFIER': ' '.join(words),
        'CONTEXT_NUMBER': context_to_number(identifier_context),
    })

    # Build features
    data = createFeatures(
        data,
        mutable_feature_list,
        modelGensimEnglish=app.model_data.ModelGensimEnglish,
    )

    # Convert any categorical features to numeric
    categorical_features = ['NLTK_POS', 'PREV_POS', 'NEXT_POS']
    for category_column in categorical_features:
        if category_column in data.columns:
            data[category_column] = data[category_column].astype(str)
            unique_vals = data[category_column].unique()
            category_map = {}
            for val in unique_vals:
                if val in universal_to_custom:
                    category_map[val] = custom_to_numeric[universal_to_custom[val]]
                else:
                    category_map[val] = custom_to_numeric['NOUN']
            data[category_column] = data[category_column].map(category_map)

    # Load classifier and annotate
    clf = joblib.load(os.path.join(SCRIPT_DIR, '..', 'models', 'model_GradientBoostingClassifier.pkl'))
    predicted_tags = annotate_identifier(clf, data)

    result = { "words": [] }
    for i, word in enumerate(words):
        dictionary = dictionary_lookup(word)
        result["words"].append({
            word: { "tag": predicted_tags[i], "dictionary": dictionary }
        })

    tag_time = time.perf_counter() - start_time
    if cache_id is not None:
        cache.add(identifier_name, result, identifier_context, tag_time)

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
    
    # Make predictions
    y_pred = clf.predict(df_features)
    return y_pred

