import os
import sqlite3
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import classifier_multiclass
import gensim.downloader as api

# import classifier_training_set_generator

input_file = 'input/revision_training_db.db'
sql_statement = 'select * from training_set_cp_minor order by ID';
# sql_statement = 'select * from training_set_conj_other order by random()';
# sql_statement = 'select * from training_set_norm order by random()';
# sql_statement = 'select * from training_set_norm_other order by random()';
identifier_column = "ID"
# independent_variables = ['WORD', 'POSITION', 'MAXPOSITION', 'NORMALIZED_POSITION', 'CONTEXT']
# independent_variables = ['TYPE', 'WORD', 'SWUM_TAG', 'POSSE_TAG', 'STANFORD_TAG', 'NORMALIZED_POSITION', 'CONTEXT']
independent_variables_base = ['NORMALIZED_POSITION']
dependent_variable = 'CORRECT_TAG'
vector_size = 300

#Conjunctions and determiners are closed set words, so we can soft-code them by doing a lookup on their
#Word embeddings. This avoids the problem with hard-coding (i.e., assuming the word is always a closet set word)
#while still giving our approach the ability to determine if we're in the most-likely context of them being a closed set word
conjunctions = ["for", "and", "nor", "but", "or", "yet", "so", "although", "after", "before", "because", "how",
                "if", "once", "since", "until", "unless", "when"]

determiners = ["a", "all", "an", "another", "any", "anybody", "anyone", "anywhere",
               "each", "either", "enough", "everybody", "everyone", "everything", "everywhere",
               "every", "first", "few", "fewer", "fewest", "hers", "his", "last", "least",
               "many", "much", "my", "neither", "next", "nobody", "none", "nothing", "nowhere", "once", "our",
               "second", "some", "somebody", "something", "somewhere", "that", "the", "their", "these", "this", "those",
               "various", "whatever", "which", "whichever", "whose", "your"]


def read_input(sql, conn):
    input_data = pd.read_sql_query(sql, conn)
    print(" --  --  --  -- Read " + str(len(input_data)) + " input rows --  --  --  -- ")
    createFeatures(input_data)
    return input_data


independent_variables_add = [[]]
independent_variables_add[0] += ["LAST_LETTER", 'CONTEXT', 'MAXPOSITION',"DIGITS", 'POSITION']

for i in range(0, vector_size):
    independent_variables_add[0].append("VEC" + str(i))

def createFeatures(data):
    startTime = time.time()
    model = createModel()
    createWordVectorsFeature(model, data)
    createLetterFeature(data)
    createDigitFeature(data)
    createDeterminerFeature(data)
    createConjunctionFeature(data)
    createFrequencyFeature(data)
    wordLength(data)
    print("Total Feature Time: " + str((time.time() - startTime)))


def wordLength(data):
    words = data["WORD"]
    wordLengths = [len(word) for word in words]
    data.insert(0, "WORD_LENGTH", wordLengths)


def createFrequencyFeature(data):
    words = data["WORD"]
    frequency = {}
    for word in words:
        word = word.lower()
        if word in frequency:
            frequency[word] = frequency[word] + 1
        else:
            frequency[word] = 1
    frequencyList = [frequency[word.lower()] for word in words]
    data.insert(0, "FREQUENCY", frequencyList)


def createConjunctionFeature(data):
    words = data["WORD"]
    isConjunction = [1 if word in conjunctions else 0 for word in words]
    data.insert(0, "CONJUNCTION", isConjunction)


def createDeterminerFeature(data):
    words = data["WORD"]
    isDeterminer = [1 if word in determiners else 0 for word in words]
    data.insert(0, "DETERMINER", isDeterminer)


def createDigitFeature(data):
    words = data["WORD"]
    isDigits = [1 if word.isdigit() else 0 for word in words]
    data.insert(0, "DIGITS", isDigits)


def createLetterFeature(data):
    lastLetters = np.array([ord(word[len(word) - 1].lower()) for word in data["WORD"]])
    data.insert(0, "LAST_LETTER", lastLetters)

def createModel(pklFile=""):
    if pklFile != "":
        model = joblib.load(pklFile)
        return model
    modelGensim = api.load('fasttext-wiki-news-subwords-300')  # CJ, D, P, VM?

    if not os.path.exists("output"):
        os.makedirs("output")
    joblib.dump(modelGensim, "output/gensimTrainingModel.pkl")

    return modelGensim

#Get word vectors for our closed set words
def createWordVectorsFeature(model, data):
    words = data["WORD"]
    zeroVector = np.zeros_like(model.get_vector("and"))
    vectors = [model.get_vector(word) if word in model.index_to_key else zeroVector for word in words]
    for i in range(0, vector_size):
        data.insert(i, "VEC" + str(i), np.array([vector[i] for vector in vectors]))
    return vectors


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
        results_text_file.write("Features: {number}. {features}\n".format(features=feature_list, number=count))
        algorithms = [classifier_multiclass.Algorithm.DECISION_TREE]
        for index in range(1):
            trainingSeed = 236373
            # trainingSeed = round(random.random() * 1000000)
            classifier_multiclass.perform_classification(df_features, df_class, text_column, results_text_file,
                                                         'output',
                                                         algorithms, trainingSeed)
            print("Run #" + str(index))
            print("Time Stamp: " + str(time.time() - intervalStart))
            print("Training Seed: " + str(trainingSeed))
            intervalStart = time.time()

        end = time.time()
        print("Process completed in " + str(end - start) + " seconds")


def annotate_word(normalized_length, code_context, last_letter, max_position, digits, position,
                  determiner, conjunction, frequency, vectors):
    input_model = 'output/model_DecisionTreeClassifier.pkl'

    data = {'NORMALIZED_POSITION': [normalized_length],
            'LAST_LETTER': [last_letter],
            'MAXPOSITION': [max_position],
            'DIGITS': [digits],
            'POSITION': [position],
            'CONTEXT': [code_context],
            }
    for i, vector in enumerate(vectors):
        data["VEC" + str(i)] = vector

    df_features = pd.DataFrame(data,
                               columns=independent_variables_base + independent_variables_add[0])

    clf = joblib.load(input_model)
    y_pred = clf.predict(df_features)
    return y_pred[0]


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
    createFeatures(df_input)
    print(df_input.head())
    for i, row in df_input.iterrows():
        actual_word = row['WORD']
        actual_identifier = row['IDENTIFIER']
        actual_pattern = row['GRAMMAR_PATTERN']
        normalized_length = row['NORMALIZED_POSITION']
        code_context = row['CONTEXT']
        correct_tag = row['CORRECT_TAG']
        system = row['SYSTEM']
        ident = row['IDENTIFIER_CODE']
        last_letter = row['LAST_LETTER']
        max_position = row['MAXPOSITION']
        digits = row['DIGITS']
        position = row['POSITION']
        determiner = row["DETERMINER"]
        conjunction = row["CONJUNCTION"]
        frequency = row["FREQUENCY"]
        vectors = []
        for i in range(vector_size):
            vectors.append(row["VEC" + str(i)])

        result = annotate_word(normalized_length, code_context, last_letter, max_position, digits, position,
                               determiner, conjunction, frequency, vectors)
        print(
            "{identifier},{pattern},{word},{correct},{prediction},{agreement},{system_name},{context},{ident}"
            .format(identifier=actual_identifier, word=actual_word, pattern=actual_pattern,
                    correct=correct_tag, prediction=result, agreement=(correct_tag == result),
                    system_name=system, context=code_context, ident=ident, last_letter=last_letter,
                    max_position=max_position, digits=digits, position=position),
            file=open(outputFile, "a"))


if __name__ == "__main__":
    # read_from_database()
    main()
