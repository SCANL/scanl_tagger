import os, sqlite3, classifier_multiclass, random, nltk, argparse
from datetime import datetime
import classifier_multiclass
from feature_generator import *
import pandas as pd
import numpy as np
from tag_identifier import start_server
from download_code2vec_vectors import *
from create_models import createModel, stable_features, mutable_feature_list

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def read_input(sql, features, conn):
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
    print(input_data.columns)
    input_data_copy = input_data.copy()
    rows = input_data_copy.values.tolist()
    random.shuffle(rows)
    shuffled_input_data = pd.DataFrame(rows, columns=input_data.columns)
    modelTokens, modelMethods, modelGensimEnglish = createModel(rootDir=SCRIPT_DIR)
    input_data = createFeatures(shuffled_input_data, features, modelGensimEnglish=modelGensimEnglish)
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
    df_input = read_input(sql_statement, independent_variables, connection)
    print(" --  -- Completed: Reading Input --  -- ")
    # ###############################################################
    
    # Create an explicit copy to avoid SettingWithCopyWarning
    df_features = df_input[independent_variables].copy()
    df_class = df_input[[dependent_variable]].copy()
    
    category_variables = []
    categorical_columns = ['NLTK_POS', 'PREVIOUS_NLTK_POS', 'TYPE', 'LANGUAGE']
    
    # Safely handle categorical variables
    for category_column in categorical_columns:
        if category_column in df_features.columns:
            category_variables.append(category_column)
            df_features.loc[:, category_column] = df_features[category_column].astype(str)
    
    # Ensure output directories exist
    output_dir = os.path.join(SCRIPT_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, 'results.txt')
    mode = 'a' if os.path.exists(filename) else 'w'
   
    with open(filename, mode) as results_text_file:
        results_text_file.write(datetime.now().strftime("%H:%M:%S") + "\n")
       
        # Print config in a readable fashion
        results_text_file.write("Configuration:\n")
        for key, value in config.items():
            results_text_file.write(f"{key}: {value}\n")
        results_text_file.write("\n")

        for category_column in category_variables:
            # Explicitly handle categorical conversion
            unique_values = df_features[category_column].unique()
            category_map = {}
            for value in unique_values:
                if value in universal_to_custom:
                    category_map[value] = custom_to_numeric[universal_to_custom[value]]
                else:
                    category_map[value] = custom_to_numeric['NM']  # Assign 'NM' (8) for unknown categories

            df_features.loc[:, category_column] = df_features[category_column].map(category_map)
       
        print(" --  -- Distribution of labels in corpus --  -- ")
        print(df_class[dependent_variable].value_counts())
        results_text_file.write(f"SQL: {sql_statement}\n")
        results_text_file.write(f"Features: {df_features}\n")
       
        algorithms = [classifier_multiclass.TrainingAlgorithm.XGBOOST]
        classifier_multiclass.perform_classification(df_features, df_class, results_text_file,
                                                    output_dir, algorithms, trainingSeed,
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
        config = {
            'input_file': os.path.join(SCRIPT_DIR, 'input', 'scanl_tagger_training_db_11_29_2024.db'),
            'sql_statement': 'select * from training_set',
            'identifier_column': "ID",
            'dependent_variable': 'CORRECT_TAG',
            'pyrandom_seed': random.randint(0, 2**32 - 1),
            'trainingSeed': random.randint(0, 2**32 - 1),
            'classifierSeed': random.randint(0, 2**32 - 1),
            'npseed': random.randint(0, 2**32 - 1),
            'independent_variables': stable_features + mutable_feature_list
        }
        train(config)
    else:
        parser.print_usage()