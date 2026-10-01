import os
import time
from flask import Flask, abort, request
from waitress import serve
from spiral import ronin
import json
import sqlite3
from src.lm_based_tagger.distilbert_tagger import DistilBertTagger
from src.tagging_backend import load_english_words

app = Flask(__name__)

lm_model = None


def _parse_optional_bool(value):
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _load_runtime_config(config_path: str | None) -> dict:
    if not config_path:
        return {}

    with open(config_path) as data:
        return json.load(data)

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

def initialize_model(temp_config = {}, runtime_config = None):
    """
    Load the DistilBERT+CRF tagger named by the startup overrides or the runtime config.
    """
    global lm_model
    runtime_config = runtime_config or {}
    print("Loading DistilBERT tagger...")
    model_path = temp_config.get("model", runtime_config.get("model"))
    if not model_path:
        raise ValueError("Run mode requires a model path or repo id.")

    is_local = temp_config.get("local")
    if is_local is None:
        is_local = bool(runtime_config.get("local", False))

    pattern_postprocessing = temp_config.get("pattern_postprocessing")
    if pattern_postprocessing is None:
        pattern_postprocessing = runtime_config.get("pattern_postprocessing")

    model_source = "local directory" if is_local else "HuggingFace repo"
    print(f"LM model source: {model_source}: {model_path}")

    lm_model = DistilBertTagger(
        model_path,
        local=is_local,
        pattern_postprocessing=pattern_postprocessing,
    )
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
    config_path = temp_config.get("config_path")
    runtime_config = _load_runtime_config(config_path)
    initialize_model(temp_config, runtime_config=runtime_config)

    print("loading cache...")
    if not os.path.isdir("cache"): os.mkdir("cache")

    print("loading dictionary")
    app.english_words = load_english_words()

    #insert english words from words/en.txt
    if not os.path.exists("words/en.txt"):
        print("could not find English words, using WordNet only!")
    else:
        with open("words/en.txt") as words:
            for word in words:
                app.english_words.add(word[:-1])

    print('retrieving server configuration...')
    config = runtime_config

    server_host = temp_config["address"] if "address" in temp_config.keys() else config.get("address", "0.0.0.0")
    server_port = temp_config["port"] if "port" in temp_config.keys() else config.get('port', 5000)
    server_url_scheme = temp_config["protocol"] if "protocol" in temp_config.keys() else config.get("protocol", "http")

    print("loading word list...")
    wordListPath = temp_config["words"] if "words" in temp_config.keys() else config.get("words", "")
    app.words = WordList(wordListPath)
    app.words.load()

    print("Starting server...")
    serve(app, host=server_host, port=server_port, url_scheme=server_url_scheme)

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

    # 2) Tag the tokens with the DistilBERT+CRF model
    result = { "words": [] }
    request_postprocessing = request.args.get("pattern_postprocessing")
    try:
        postprocessing_override = _parse_optional_bool(request_postprocessing)
    except ValueError as exc:
        abort(400, description=str(exc))

    tags = lm_model.tag_identifier(
        tokens=words,
        context=identifier_context,
        type_str=data_type,
        language=programming_language,
        system_name=system_name,
        pattern_postprocessing=postprocessing_override,
    )

    for word, tag in zip(words, tags):
        dictionary = dictionary_lookup(word)
        result["words"].append({
            word: { "tag": tag, "dictionary": dictionary }
        })

    tag_time = time.perf_counter() - start_time
    if cache is not None:
        cache.add(identifier_name, result, identifier_context, tag_time)
    return result
