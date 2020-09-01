import os
import sqlite3
import time

import numpy as np
import pandas as pd
import classifier_multiclass

start = time.time()

input_file = 'input/input2.sqlite'
sql_statement = 'select * from combined_data_other order by random()';
identifier_column = "ID"
text_column = 'WORD'
#independent_variables = ['WORD', 'POSITION', 'MAXPOSITION', 'NORMALIZED_POSITION', 'CONTEXT']
independent_variables = ['SWUM_TAG', 'POSSE_TAG', 'STANFORD_TAG', 'NORMALIZED_POSITION', 'CONTEXT']
category_variables =['SWUM_TAG', 'POSSE_TAG', 'STANFORD_TAG']
dependent_variable = 'CORRECT_TAG'

def read_input(sql, conn):
    input_data = pd.read_sql_query(sql, conn)
    print(" --  --  --  -- Read " + str(len(input_data)) + " input rows --  --  --  -- ")
    return input_data



def main():
    start = time.time()

    # ###############################################################
    print(" --  -- Started: Reading Database --  -- ")
    connection = sqlite3.connect(input_file)
    df_input = read_input(sql_statement, connection)
    print(" --  -- Completed: Reading Input --  -- ")
    # ###############################################################

    df_input.set_index(identifier_column, inplace=True)
    df_features = df_input[independent_variables]
    df_class = df_input[[dependent_variable]]

    for category_column in category_variables:
        if category_column in df_features.columns:
            df_features[category_column] = df_features[category_column].astype('category')
            df_features[category_column] = df_features[category_column].cat.codes


    print(" --  -- Distribution of labels in corpus --  -- ")
    print(df_class[dependent_variable].value_counts())

    if not os.path.exists('output'):
        os.makedirs('output')
    filename = 'output/results.txt'
    if os.path.exists(filename):
        append_write = 'a'
    else:
        append_write = 'w'
    results_text_file = open(filename, append_write)

    results_text_file.write("Features: %s" % independent_variables)
    classifier_multiclass.perform_classification(df_features, df_class, text_column, results_text_file, 'output')


    end = time.time()
    print("Process completed in " + str(end - start) + " seconds")




if __name__ == "__main__":
    main()