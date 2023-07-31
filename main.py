import os
import sqlite3
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import classifier_multiclass
import gensim.downloader as api
import random
import nltk
# import classifier_training_set_generator

input_file = 'input/det_conj_db.db'
sql_statement = 'select * from base order by random()'
# sql_statement = 'select * from training_set_conj_other order by random()';
# sql_statement = 'select * from training_set_norm order by random()';
# sql_statement = 'select * from training_set_norm_other order by random()';
identifier_column = "ID"
# independent_variables = ['WORD', 'POSITION', 'MAXPOSITION', 'NORMALIZED_POSITION', 'CONTEXT']
# independent_variables = ['TYPE', 'WORD', 'SWUM_TAG', 'POSSE_TAG', 'STANFORD_TAG', 'NORMALIZED_POSITION', 'CONTEXT']
independent_variables_base = ['NORMALIZED_POSITION']
dependent_variable = 'CORRECT_TAG'
vector_size = 300
trainingSeed = 236373
#Conjunctions and determiners are closed set words, so we can soft-code them by doing a lookup on their
#Word embeddings. This avoids the problem with hard-coding (i.e., assuming the word is always a closet set word)
#while still giving our approach the ability to determine if we're in the most-likely context of them being a closed set word
conjunctions = ["for", "and", "nor", "but", "or", "yet", "so", "although", "after", "before", "because", "how",
                "if", "once", "since", "until", "unless", "when", "as", "that", "though", "till", "while", "where", "after",
                "although", "as", "as if", "as long as", "as much as", "as soon as", "as far as", "as though", "by the time",
                "in as much as", "in as much", "in order to", "in order that", "in case", "lest", "though", "now that", "now since", 
                "now when", "now", "even if", "even", "even though", "provided", "provided that", "else", "if then", "if when", "if only", 
                "just as", "where", "wherever", "whereas", "where if", "whether", "since", "because", "whose", "whoever", "unless",
                "while", "before", "why", "so that", "until", "how", "since", "than", "till", "whenever", "supposing", "when", 
                "or not", "what", "also", "otherwise", "for", "neither nor", "and", "not only but also", "nor", "whether or", "but", 
                "so that", "or", "such that", "yet", "as soon as", "so", "as well as", "also", "provided that", "as well as", "whoever", 
                "yet", "while", "still", "until", "too", "unless", "only", "since", "however", "as if", "no less than", "no less than", 
                "which", "otherwise", "where", "in order that", "who", "than", "after", "as", "because", "either or", "whoever", "nevertheless", 
                "though", "else", "although", "if", "if", "while", "till", "no sooner than"]

determiners = ["a", "a few", "a little", "all", "an", "another", "any", "anybody", "anyone", "anything", "anywhere", "both", "certain", "each", 
               "either", "enough", "every", "everybody", "everyone", "everything", "everywhere", "few", "fewer", "fewest", "last", "least", "less", 
               "little", "many", "many a", "more", "most", "much", "neither", "next", "no", "no one", "nobody", "none", "nothing", "nowhere", "once", 
               "said", "several", "some", "somebody", "something", "somewhere", "sufficient", "that", "the", "these", "this", "those", "us", 
               "various", "we", "what", "whatever", "which", "whichever", "you"]

independent_variables_add = [[]]
independent_variables_add[0] += ["LAST_LETTER", 'CONTEXT', 'WORD LENGTH', 'MAXPOSITION', 'NLTK_POS']

for i in range(0, vector_size):
    independent_variables_add[0].append("VEC" + str(i))

def createFeatures(data):
    startTime = time.time()
    model = createModel()
    data = createWordVectorsFeature(model, data)
    data = createLetterFeature(data)
    data = createDigitFeature(data)
    data = createDeterminerFeature(data)
    data = createConjunctionFeature(data)
    data = createFrequencyFeature(data)
    data = wordLength(data)
    data = maxPosition(data)
    data = wordPosTag(data)
    print("Total Feature Time: " + str((time.time() - startTime)))
    return data

def wordPosTag(data):
    words = data["WORD"]
    pos_tags = pd.DataFrame([nltk.pos_tag([word])[-1][-1] for word in words])
    pos_tags.columns = ['NLTK_POS']
    data = pd.concat([data, pos_tags], axis=1)
    return data


def wordLength(data):
    words = data["WORD"]
    wordLengths = pd.DataFrame([len(word) for word in words])
    wordLengths.columns = ['WORD LENGTH']
    data = pd.concat([data, wordLengths], axis=1)
    return data

def maxPosition(data):
    identifiers = data["IDENTIFIER"]
    maxPosition = pd.DataFrame([len(identifier) for identifier in identifiers])
    maxPosition.columns = ['MAXPOSITION']
    data = pd.concat([data, maxPosition], axis=1)
    return data

def createFrequencyFeature(data):
    words = data["WORD"]
    frequency = {}
    for word in words:
        word = word.lower()
        if word in frequency:
            frequency[word] = frequency[word] + 1
        else:
            frequency[word] = 1
    frequencyList = pd.DataFrame([frequency[word.lower()] for word in words])
    frequencyList.columns = ['FREQUENCY']
    data = pd.concat([data, frequencyList], axis=1)
    return data


def createConjunctionFeature(data):
    words = data["WORD"]
    isConjunction = pd.DataFrame([1 if word in conjunctions else 0 for word in words])
    isConjunction.columns = ["CONJUNCTION"]
    data = pd.concat([data, isConjunction], axis=1)
    return data


def createDeterminerFeature(data):
    words = data["WORD"]
    isDeterminer = pd.DataFrame([1 if word in determiners else 0 for word in words])
    isDeterminer.columns = ["DETERMINER"]
    data = pd.concat([data, isDeterminer], axis=1)
    return data


def createDigitFeature(data):
    words = data["WORD"]
    isDigits = pd.DataFrame([1 if word.isdigit() else 0 for word in words])
    isDigits.columns = ["DIGITS"]
    data = pd.concat([data, isDigits], axis=1)
    return data


def createLetterFeature(data):
    lastLetters = pd.DataFrame(np.array([ord(word[len(word) - 1].lower()) for word in data["WORD"]]))
    lastLetters.columns = ["LAST_LETTER"]
    data = pd.concat([data, lastLetters], axis=1)
    return data

#Get word vectors for our closed set words
def createWordVectorsFeature(model, data):
    words = data["WORD"]
    zeroVector = np.zeros_like(model.get_vector("and"))
    vectors = [model.get_vector(word) if word in model.index_to_key else zeroVector for word in words]
    cnames = [f'VEC{i}' for i in range(0, vector_size)]
    df = pd.DataFrame()
    for i in range(0, vector_size):
        df = pd.concat([df, pd.DataFrame([vector[i] for vector in vectors])], axis=1)
    df.columns = cnames

    data = pd.concat([data, df], axis=1)
    return data

def createModel(pklFile=""):
    if pklFile != "":
        model = joblib.load(pklFile)
        return model
    modelGensim = api.load('fasttext-wiki-news-subwords-300')  # CJ, D, P, VM?

    if not os.path.exists("output"):
        os.makedirs("output")
    joblib.dump(modelGensim, "output/gensimTrainingModel.pkl")

    return modelGensim

def read_input(sql, conn):
    input_data = pd.read_sql_query(sql, conn)
    print(" --  --  --  -- Read " + str(len(input_data)) + " input rows --  --  --  -- ")
    input_data = createFeatures(input_data)
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
        algorithms = [classifier_multiclass.Algorithm.DECISION_TREE]
        for index in range(1):
            classifier_multiclass.perform_classification(df_features, df_class, text_column, results_text_file,
                                                         'output',
                                                         algorithms, trainingSeed)
            print("Run #" + str(index))
            print("Time Stamp: " + str(time.time() - intervalStart))
            print("Training Seed: " + str(trainingSeed))
            intervalStart = time.time()

        end = time.time()
        print("Process completed in " + str(end - start) + " seconds")


############CURRENTLY NOT EXECUTED###############

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
    input_file = 'input/conjunctiondb.db'
    sql_statement = "select * from base"
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
    print("DF")
    print(df_input)
    print("DF END")
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
