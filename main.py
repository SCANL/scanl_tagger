import os, sqlite3, classifier_multiclass, random, nltk, argparse
from datetime import datetime
import classifier_multiclass
from feature_generator import *
import pandas as pd
import numpy as np
from tag_identifier import start_server
from download_code2vec_vectors import *

def read_input(sql, conn):
    """
    Read input data from an SQLite database and preprocess it.

    This function reads data from the specified SQL query and database connection, shuffles the rows, and then applies
    a preprocessing function called 'createFeatures' to create additional features.

    Args:
        sql (str): The SQL query to fetch data from the database.
        conn (sqlite3.Connection): The SQLite database connection.

    Returns:
        pandas.DataFrame: A DataFrame containing the preprocessed input data.
    """
    input_data = pd.read_sql_query(sql, conn)
    print(" --  --  --  -- Read " + str(len(input_data)) + " input rows --  --  --  -- ")

    input_data_copy = input_data.copy()
    rows = input_data_copy.values.tolist()
    random.shuffle(rows)
    shuffled_input_data = pd.DataFrame(rows, columns=input_data.columns)

    input_data = createFeatures(shuffled_input_data)
    return input_data

def train(config):
    """
    Train a part of speech tagger model using specified features and a training dataset.
    This function reads data from an SQLite database, preprocesses it, and performs classification using a specified set
    of features. The results are written to an output file, including information about the training process and the
    distribution of labels in the training data.
    Args:
        config (dict): A dictionary containing configuration data.
    Returns:
        None
    """
    nltk.download('universal_tagset')
   
    # Extract configuration values from the 'config' dictionary
    input_file = config['input_file']
    sql_statement = config['sql_statement']
    identifier_column = config['identifier_column']
    dependent_variable = config['dependent_variable']
    pyrandom_seed = config['pyrandom_seed']
    trainingSeed = config['trainingSeed']
    classifierSeed = config['classifierSeed']
   
    np.random.seed(config['npseed'])
    random.seed(pyrandom_seed)
    independent_variables = config['independent_variables']
    # ###############################################################
    print(" --  -- Started: Reading Database --  -- ")
    connection = sqlite3.connect(input_file)
    df_input = read_input(sql_statement, connection)
    print(" --  -- Completed: Reading Input --  -- ")
    # ###############################################################
    category_variables = []
    feature_list = independent_variables
    df_input.set_index(identifier_column, inplace=True)
    df_features = df_input[feature_list]
    if 'NLTK_POS' in feature_list:
        category_variables.append('NLTK_POS')
        df_features['NLTK_POS'] = df_features['NLTK_POS'].astype(str)
    if 'PREVIOUS_NLTK_POS' in feature_list:
        category_variables.append('PREVIOUS_NLTK_POS')
        df_features['PREVIOUS_NLTK_POS'] = df_features['PREVIOUS_NLTK_POS'].astype(str)
    if 'TYPE' in feature_list:
        category_variables.append('TYPE')
        df_features['TYPE'] = df_features['TYPE'].astype(str)
    df_class = df_input[[dependent_variable]]
    if not os.path.exists('output'):
        os.makedirs('output')
    filename = 'output/results.txt'
    mode = 'a' if os.path.exists(filename) else 'w'
    
    with open(filename, mode) as results_text_file:
        results_text_file.write(datetime.now().strftime("%H:%M:%S") + "\n")
        
        # Print config in a readable fashion
        results_text_file.write("Configuration:\n")
        for key, value in config.items():
            results_text_file.write(f"{key}: {value}\n")
        results_text_file.write("\n")
        
        for category_column in category_variables:
            if category_column in df_features.columns:
                df_features[category_column] = df_features[category_column].astype('category')
                d = dict(enumerate(df_features[category_column].cat.categories))
                results_text_file.write(f"{category_column}: {d}\n")
                df_features[category_column] = df_features[category_column].cat.codes
        
        print(" --  -- Distribution of labels in corpus --  -- ")
        print(df_class[dependent_variable].value_counts())
        results_text_file.write(f"SQL: {sql_statement}\n")
        results_text_file.write(f"Features: {df_features}\n")
        
        algorithms = [classifier_multiclass.TrainingAlgorithm.RANDOM_FOREST]
        classifier_multiclass.perform_classification(df_features, df_class, results_text_file,
                                                    'output', algorithms, trainingSeed,
                                                    classifierSeed)

if __name__ == "__main__":
    """
    Use argparse to allow the user to choose either running the tagger or training a new tagger

    Usage:
    - To check the application version, use: -v or --version.
    - To start a server for part-of-speech tagging requests, use: -r or --run.
    - To run a training set and retrain the model, use: -t or --train.

    Example Usage:
    python script.py -v  # Display the application version.
    python script.py -r  # Start the server for tagging requests.
    python script.py -t  # Run the training set to retrain the model.

    Note:
    If no arguments are provided or if there is an invalid argument, the script will display usage instructions.

    Author: Christian Newman
    Version: 1.5.0
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--version", action="store_true", help="print tagger application version")
    parser.add_argument("-r", "--run", action="store_true", help="run server for part of speech tagging requests") 
    parser.add_argument("-t", "--train", action="store_true", help="run training set to retrain the model")

    args = parser.parse_args()

    if args.version:
        print("SCANL Tagger version 1.5.0")
    elif args.run:
        download_files()
        start_server()
    elif args.train:
        download_files()
        # Define a configuration dictionary and pass it to the train function
        # 'CONJUNCTION'
        config = {
            'input_file': 'input/scanl_tagger_training_db_8_29_2024.db',
            'sql_statement': 'select * from training_set',
            'identifier_column': "ID",
            'dependent_variable': 'CORRECT_TAG',
            'pyrandom_seed': random.randint(0, 2**32 - 1),
            'trainingSeed': random.randint(0, 2**32 - 1),
            'classifierSeed': random.randint(0, 2**32 - 1),
            'npseed': random.randint(0, 2**32 - 1),
            'independent_variables': ['NORMALIZED_POSITION', 'LAST_LETTER', 'CONTEXT_NUMBER', 'MAXPOSITION',
                                      'NLTK_POS', 'POSITION', 'VERB_SCORE', 'DET_SCORE', 'PREP_SCORE',
                                      'CONJ_SCORE', 'PREPOSITION', 'DETERMINER', 'ENGLISHV_SCORE',
                                      'ENGLISHN_SCORE', 'METHODN_SCORE', 'METHODV_SCORE', 'CODEPRE_SCORE',
                                      'METHODPRE_SCORE', 'ENGLISHPRE_SCORE', 'CONTAINSDIGIT', 'CONTAINSCLOSEDSET', 'SECOND_LAST_LETTER', 'NOUN_SCORE']
        }
        train(config)
    else:
        parser.print_usage()
