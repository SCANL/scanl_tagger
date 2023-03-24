import math
import os
import sqlite3
import time
from datetime import datetime
import random

import joblib
import numpy as np
import pandas as pd
from gensim.models.word2vec_inner import REAL
from pandas import DataFrame

import classifier_multiclass
from gensim.models import Word2Vec
from nltk.tokenize import WordPunctTokenizer
from numpy.linalg import norm
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
                                 # "DIGITS", 'POSITION', "DETERMINER", "CONJUNCTION", "FREQUENCY"]
# 'WORD'


for i in range(0, vector_size):
    independent_variables_add[0].append("VEC" + str(i))
# "DETERMINER","CONJUNCTION","FREQUENCY", 'CONTEXT','WORD',

def createFeatures(data):
    print(data)
    startTime = time.time()
    words = data["WORD"]
    model = createModel(words, "output/gensimTrainingModel.pkl")
    vectors = createWordVectorsFeature(model, data)
    # createCosinesFeature(model, data, vectors)
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


def getCosine(vectorA, vectorB):
    cosine = np.dot(vectorA, vectorB) / (norm(vectorA, axis=0) * norm(vectorB))
    if math.isnan(cosine):
        return 0
    else:
        return cosine


def createModel(words, pklFile=""):
    if pklFile != "":
        model = joblib.load(pklFile)
        return model
    modelGensim = api.load('fasttext-wiki-news-subwords-300')  # CJ, D, P, VM?
    # for word in words[0:20]:
    #     print(word, modelGensim.most_similar(word.lower()))
    # modelWords = list(words) + list(modelGensim.index_to_key)
    #
    # # '/Users/gavinburris/gensim-data/conceptnet-numberbatch-17-06-300/conceptnet-numberbatch-17-06-300.gz
    # # '/Users/gavinburris/gensim-data/glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz
    # # '/Users/gavinburris/gensim-data/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz'
    # # '/Users/gavinburris/gensim-data/glove-twitter-200/glove-twitter-200.gz' Had 1.00 Recall for VM
    # # '/Users/gavinburris/gensim-data/fasttext-wiki-news-subwords-300/fasttext-wiki-news-subwords-300.gz'
    #
    # tokenizer = WordPunctTokenizer()
    # words_tok = [tokenizer.tokenize(word.lower()) for word in modelWords]
    # model = Word2Vec(words_tok, vector_size=vector_size, min_count=1, workers=1).wv
    # model.vectors_lockf = np.ones(len(model), dtype=REAL)
    joblib.dump(modelGensim, "output/gensimTrainingModel.pkl")
    # model.save("output/gensimTrainingModel.pkl")

    # model.intersect_word2vec_format(
    #     '/Users/gavinburris/gensim-data/conceptnet-numberbatch-17-06-300/conceptnet-numberbatch-17-06-300.gz')
    # for word in words[0:20]:
    #     print(word, model.most_similar(word.lower()))
    return modelGensim


def createWordVectorsFeature(model, data):
    words = data["WORD"]
    zeroVector = np.zeros_like(model.get_vector("and"))
    vectors = [model.get_vector(word) if word in model.index_to_key else zeroVector for word in words]
    for i in range(0, vector_size):
        data.insert(i, "VEC" + str(i), np.array([vector[i] for vector in vectors]))
    return vectors


def createCosinesFeature(model, data, vectors):
    words = data["WORD"]
    pos = data["CORRECT_TAG"]
    # posDict = {"CJ": [], "D": [], "DT": [], "N": [], "NM": [], "NPL": [], "P": [], "PRE": [], "V": [], "VM": []}
    digits = [digit for digit in range(0, 100)]
    posDict = {"CJ": conjunctions, "D": digits, "DT": determiners, "P": [], "PRE": []}
    cosinesCJ = []
    cosinesD = []
    cosinesDT = []

    zeroVector = [0] * 300
    # for i in range(len(words)):
    #     posDict[pos[i]].append(words[i].lower())
    # print(posDict["P"])
    # print(posDict["PRE"])

    for key in posDict.keys():
        # if word in model.index_to_key else zeroVector
        posDict[key] = np.array([model.get_vector(word) for word in
                                 posDict[key]]).mean(axis=0)

    for i in range(len(words)):
        print(words[i], getCosine(vectors[i], posDict["CJ"]), getCosine(vectors[i], posDict["D"]), getCosine(vectors[i], posDict["DT"]))
        cosinesCJ.append(getCosine(vectors[i], posDict["CJ"]))
        cosinesD.append(getCosine(vectors[i], posDict["D"]))
        cosinesDT.append(getCosine(vectors[i], posDict["DT"]))

    print("COSINE_CJ", cosinesCJ)
    print("COSINE_D", cosinesD)
    print("COSINE_DT", cosinesDT)
    data.insert(0, "COSINE_CJ", np.array(cosinesCJ))
    data.insert(0, "COSINE_D", np.array(cosinesD))
    data.insert(0, "COSINE_DT", np.array(cosinesDT))


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
            # 'DETERMINER': [determiner],
            # 'CONJUNCTION': [conjunction],
            # 'FREQUENCY': [frequency]
            }
    for i, vector in enumerate(vectors):
        data["VEC" + str(i)] = vector

    df_features = pd.DataFrame(data,
                               columns=independent_variables_base + independent_variables_add[0])
    # df_features = pd.DataFrame(data,
    #                            columns=['SWUM_TAG', 'POSSE_TAG', 'STANFORD_TAG', 'FLAIR_TAG', 'NORMALIZED_POSITION', 'CONTEXT'])

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
    for i, row in df_input.iterrows():
        print(i)
        print(row)
        actual_word = row['WORD']
        actual_identifier = row['IDENTIFIER']
        actual_pattern = row['GRAMMAR_PATTERN']
        normalized_length = row['NORMALIZED_POSITION']
        code_context = row['CONTEXT']
        correct_tag = row['CORRECT_TAG']
        system = row['SYSTEM']
        ident = row['IDENTIFIER_CODE']
        last_letter = row['LAST_LETTER']
        # word_type = row['TYPE']
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
