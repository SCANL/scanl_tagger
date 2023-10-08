import os, sqlite3, classifier_multiclass, random, nltk, argparse
from datetime import datetime
import classifier_multiclass
from feature_generator import *
import pandas as pd
import numpy as np
from tag_identifier import start_server

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

def train():
    """
    Train a part of speech tagger model using specified features and a training dataset.

    This function reads data from an SQLite database, preprocesses it, and performs classification using a specified set
    of features. The results are written to an output file, including information about the training process and the
    distribution of labels in the training data.

    Returns:
        None
    """

    nltk.download('universal_tagset')
    input_file = 'input/det_conj_db2.db'
    sql_statement = 'select * from base'
    identifier_column = "ID"
    independent_variables_base = ['NORMALIZED_POSITION']
    dependent_variable = 'CORRECT_TAG'
    
    seed = 1340345
    
    trainingSeed = 2227339
    classifierSeed = 3801578
    
    np.random.seed(1129175)
    random.seed(seed)

    independent_variables_add = [[]]
    independent_variables_add[0] += ["LAST_LETTER", 'CONTEXT', 'MAXPOSITION', 'NLTK_POS', 'POSITION', 'VERB_SCORE', 'DET_SCORE', 'PREP_SCORE', 'CONJ_SCORE', 'PREPOSITION', 'DETERMINER', 'ENGLISHV_SCORE', 'ENGLISHN_SCORE','METHODN_SCORE', 'METHODV_SCORE', 'CODEPRE_SCORE', 'METHODPRE_SCORE', 'ENGLISHPRE_SCORE']
    count = 0
    for feature_list in independent_variables_add:
        count = count + 1
        start = time.time()
        intervalStart = start

        # ###############################################################
        print(" --  -- Started: Reading Database --  -- ")
        connection = sqlite3.connect(input_file)
        df_input = read_input(sql_statement, connection)
        print(" --  -- Completed: Reading Input --  -- ")
        # ###############################################################

        category_variables = []
        text_column = ""

        feature_list = independent_variables_base + feature_list
        df_input.set_index(identifier_column, inplace=True)
        df_features = df_input[feature_list]
        if 'NLTK_POS' in feature_list:
            category_variables.append('NLTK_POS')
            df_features['NLTK_POS'] = df_features['NLTK_POS'].astype(str)
        if 'TYPE' in feature_list:
            category_variables.append('TYPE')
            df_features['TYPE'] = df_features['TYPE'].astype(str)

        df_class = df_input[[dependent_variable]]

        if not os.path.exists('output'):
            os.makedirs('output')
        filename = 'output/results.txt'
        if os.path.exists(filename):
            append_write = 'a'
        else:
            append_write = 'w'

        results_text_file = open(filename, append_write)
        results_text_file.write(datetime.now().strftime("%H:%M:%S") + "\n")
        for category_column in category_variables:
            if category_column in df_features.columns:
                df_features[category_column] = df_features[category_column].astype('category')
                d = dict(enumerate(df_features[category_column].cat.categories))
                results_text_file.write(str(category_column) + ":" + str(d) + "\n")
                df_features[category_column] = df_features[category_column].cat.codes

        print(" --  -- Distribution of labels in corpus --  -- ")
        print(df_class[dependent_variable].value_counts())

        results_text_file.write("SQL: %s\n" % sql_statement)
        results_text_file.write("Features: {number}. {features}\n".format(features=df_features, number=count))
        algorithms = [classifier_multiclass.Algorithm.RANDOM_FOREST]
        for index in range(1):
            classifier_multiclass.perform_classification(df_features, df_class, text_column, results_text_file,
                                                         'output',
                                                         algorithms, trainingSeed, classifierSeed)
            print("Run #" + str(index))
            print("Time Stamp: " + str(time.time() - intervalStart))
            print("Training Seed: " + str(trainingSeed))
            print("Classifier seed: " + str(classifierSeed))
            intervalStart = time.time()

        end = time.time()
        print("Process completed in " + str(end - start) + " seconds")

if __name__ == "__main__":
    """
    This script defines command-line arguments using the argparse library and performs various actions based on the provided arguments.

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
    Version: ???
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--version", action="store_true", help="print tagger application version")
    parser.add_argument("-r", "--run", action="store_true", help="run server for part of speech tagging requests") 
    parser.add_argument("-t", "--train", action="store_true", help="run training set to retrain the model")

    args = parser.parse_args()

    if args.version:
        print("App version ???")
    elif args.run:
        start_server()
    elif args.train:
        train()
    else:
        parser.print_usage()
