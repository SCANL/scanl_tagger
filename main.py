import os
import sqlite3
import time
from datetime import datetime
import numpy as np
import pandas as pd
import classifier_training_set_generator

input_file = 'input/ensemble_train_db.db'
sql_statement = 'select * from training_set_conj order by ID';
#sql_statement = 'select * from training_set_conj_other order by random()';
#sql_statement = 'select * from training_set_norm order by random()';
#sql_statement = 'select * from training_set_norm_other order by random()';
identifier_column = "ID"
#independent_variables = ['WORD', 'POSITION', 'MAXPOSITION', 'NORMALIZED_POSITION', 'CONTEXT']
#independent_variables = ['TYPE', 'WORD', 'SWUM_TAG', 'POSSE_TAG', 'STANFORD_TAG', 'NORMALIZED_POSITION', 'CONTEXT', 'ISPLURALLIST', 'ISBOOLVERB']
independent_variables_base = ['SWUM_TAG', 'POSSE_TAG', 'STANFORD_TAG']
dependent_variable = 'CORRECT_TAG'

def read_input(sql, conn):
    input_data = pd.read_sql_query(sql, conn)
    print(" --  --  --  -- Read " + str(len(input_data)) + " input rows --  --  --  -- ")
    return input_data
#[],
independent_variables_add = [['NORMALIZED_POSITION', 'CONTEXT']]

def main():
    count = 0
    for feature_list in independent_variables_add:
        count = count+1
        start = time.time()
        # ###############################################################
        print(" --  -- Started: Reading Database --  -- ")
        connection = sqlite3.connect(input_file)
        df_input = read_input(sql_statement, connection)
        print(" --  -- Completed: Reading Input --  -- ")
        # ###############################################################
        
        category_variables = ['SWUM_TAG', 'POSSE_TAG', 'STANFORD_TAG']
        text_column = ""

        feature_list = independent_variables_base + feature_list
        df_input.set_index(identifier_column, inplace=True)
        df_features = df_input[feature_list]
        if 'WORD' in feature_list:
            category_variables.append('WORD')
            df_features['WORD'] = df_features['WORD'].astype(str)
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
        results_text_file.write(datetime.now().strftime("%H:%M:%S")+"\n")
        for category_column in category_variables:
            if category_column in df_features.columns:
                df_features[category_column] = df_features[category_column].astype('category')
                d = dict(enumerate(df_features[category_column].cat.categories))
                results_text_file.write(str(category_column) + ":" + str(d)+"\n")
                df_features[category_column] = df_features[category_column].cat.codes


        print(" --  -- Distribution of labels in corpus --  -- ")
        print(df_class[dependent_variable].value_counts())

        results_text_file.write("SQL: %s\n" % sql_statement)
        results_text_file.write("Features: {number}. {features}\n".format(features=feature_list, number=count))
        classifier_training_set_generator.perform_classification(df_features, df_class, text_column, results_text_file, 'output')


        end = time.time()
        print("Process completed in " + str(end - start) + " seconds")



if __name__ == "__main__":
    main()