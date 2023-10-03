import os, sqlite3, classifier_multiclass, random, nltk
from datetime import datetime
import classifier_multiclass
from feature_generator import *
import pandas as pd
import numpy as np

# import classifier_training_set_generator
nltk.download('universal_tagset')

input_file = 'input/det_conj_db2.db'
sql_statement = 'select * from base'
# sql_statement = 'select * from training_set_conj_other order by random()';
# sql_statement = 'select * from training_set_norm order by random()';
# sql_statement = 'select * from training_set_norm_other order by random()';
identifier_column = "ID"
# independent_variables = ['WORD', 'POSITION', 'MAXPOSITION', 'NORMALIZED_POSITION', 'CONTEXT']
# independent_variables = ['TYPE', 'WORD', 'SWUM_TAG', 'POSSE_TAG', 'STANFORD_TAG', 'NORMALIZED_POSITION', 'CONTEXT']
independent_variables_base = ['NORMALIZED_POSITION']
dependent_variable = 'CORRECT_TAG'
vector_size = 128
#vector_size_e = 300

# Training Seed: 2797879, 532479
# Classifier seed: 1271197, 948572

#db2
# Training Seed: 2227339
# Classifier Seed: 3801578
# SEED: 1340345

seed = 1340345
print("SEED: " + str(seed))
trainingSeed = 2227339
classifierSeed = 3801578
np.random.seed(1129175)
random.seed(seed)

independent_variables_add = [[]]
independent_variables_add[0] += ["LAST_LETTER", 'CONTEXT', 'MAXPOSITION', 'NLTK_POS', 'POSITION', 'VERB_SCORE', 'DET_SCORE', 'PREP_SCORE', 'CONJ_SCORE', 'PREPOSITION', 'DETERMINER', 'ENGLISHV_SCORE', 'ENGLISHN_SCORE','METHODN_SCORE', 'METHODV_SCORE', 'CODEPRE_SCORE', 'METHODPRE_SCORE', 'ENGLISHPRE_SCORE'] #  'FIRST_WORD_LENGTH', 'FIRST_WORD_CAPS' 'CONJUNCTION', 'DIGITS'

def read_input(sql, conn):
    input_data = pd.read_sql_query(sql, conn)
    print(" --  --  --  -- Read " + str(len(input_data)) + " input rows --  --  --  -- ")

    input_data_copy = input_data.copy()
    rows = input_data_copy.values.tolist()
    random.shuffle(rows)
    shuffled_input_data = pd.DataFrame(rows, columns=input_data.columns)

    input_data = createFeatures(shuffled_input_data)
    return input_data

def main():
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


############CURRENTLY NOT EXECUTED###############

def read_from_database():
    input_file = 'input/revision_testing_db.db'
    sql_statement = "select * from testing_set_cp_minor"
    # sql_statement = "select * from testing_set_ca_minor"
    # sql_statement = "select * from testing_set_np_minor"
    # sql_statement = "select * from testing_set_na_minor"
    connection = sqlite3.connect(input_file)

    df_input = pd.read_sql_query(sql_statement, connection)
    outputFile = "output/model_DecisionTreeClassifier_predictions.csv"
    print(" --  --  --  -- Read " + str(len(df_input)) + " input rows --  --  --  -- ")
    print("IDENTIFIER,GRAMMAR_PATTERN,WORD,SWUM,STANFORD,CORRECT,PREDICTION,MATCH,SYSTEM,CONTEXT,IDENT",
          file=open(outputFile, "a"))
    df_input = createFeatures(df_input)
    
    category_variables = []
    if 'NLTK_POS' in df_input:
        category_variables.append('NLTK_POS')
        df_input['NLTK_POS'] = df_input['NLTK_POS'].astype(str)
    
    for category_column in category_variables:
        if category_column in df_input.columns:
            df_input[category_column] = df_input[category_column].astype('category')
            d = dict(enumerate(df_input[category_column].cat.categories))
            df_input[category_column] = df_input[category_column].cat.codes
    
    results_list = []
    start = time.time()
    for i, row in df_input.iterrows():
        actual_word = row['WORD']
        actual_identifier = row['IDENTIFIER']
        actual_pattern = row['GRAMMAR_PATTERN']
        
        params = {
            'normalized_length': row['NORMALIZED_POSITION'],
            'code_context': row['CONTEXT'],
            'last_letter': row['LAST_LETTER'],
            'max_position': row['MAXPOSITION'],
            'position': row['POSITION'],
            'determiner': row['DETERMINER'],
            'nltk_pos' : row['NLTK_POS'],
            #'conjunction': row['CONJUNCTION'],
            'verb_score': row['VERB_SCORE'],
            'det_score': row['DET_SCORE'],
            'prep_score': row['PREP_SCORE'],
            'conj_score': row['CONJ_SCORE'],
            'prep': row['PREPOSITION'],
            'det': row['DETERMINER'],
            'englishv_score': row['ENGLISHV_SCORE'],
            'englishn_score': row['ENGLISHN_SCORE'],
            'methodn_score': row['METHODN_SCORE'],
            'methodv_score': row['METHODV_SCORE'],
            'codepre_score': row['CODEPRE_SCORE'],
            'methodpre_score': row['METHODPRE_SCORE'],
            'englishpre_score': row['ENGLISHPRE_SCORE'],
            'first_word_len': row['FIRST_WORD_LENGTH'],
            'first_word_caps': row['FIRST_WORD_CAPS']
        }
        
        result = annotate_word(params)

        # Append the results to the results_list
        results_list.append({
            'identifier': actual_identifier,
            'pattern': actual_pattern,
            'word': actual_word,
            'correct': row['CORRECT_TAG'],
            'prediction': result,
            'agreement': (row['CORRECT_TAG'] == result),
            'system_name': row['SYSTEM'],
            'context': row['CONTEXT'],
            'ident': row['IDENTIFIER_CODE'],
            'last_letter': row['LAST_LETTER'],
            'max_position': row['MAXPOSITION'],
            'position': row['POSITION']
        })

    end = time.time()
    print("Process completed in " + str(end - start) + " seconds")

    results_df = pd.DataFrame(results_list)
    output_file = "output/model_RandomForestClassifier_predictions.csv"
    results_df.to_csv(output_file, index=False, mode='a')



if __name__ == "__main__":
    #read_from_database()
    main()
