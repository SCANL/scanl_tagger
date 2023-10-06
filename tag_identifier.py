import os, joblib
import pandas as pd
from feature_generator import *
from flask import Flask
from spiral import ronin

app = Flask(__name__)

class ModelData:
    def __init__(self, ModelTokens, ModelMethods, ModelGensimEnglish) -> None:
        self.ModelTokens = ModelTokens
        self.ModelMethods = ModelMethods
        self.ModelGensimEnglish = ModelGensimEnglish

def initialize_model():
    print("Loading word vectors!!")
    ModelTokens, ModelMethods, ModelGensimEnglish = createModel()
    print("Word vectors loaded!!")

    app.model_data = ModelData(ModelTokens, ModelMethods, ModelGensimEnglish)

def start_server():
    initialize_model()
    print("Starting server!!")
    app.run(host='0.0.0.0')

@app.route('/<identifier_type>/<identifier_name>/<identifier_context>')
def listen(identifier_type, identifier_name, identifier_context):
    print("INPUT: {ident_type} {ident_name} {ident_context}".format(ident_type=identifier_type, ident_name=identifier_name, ident_context=identifier_context))
    
    words = ronin.split(identifier_name)
    max_pos = [len(words) for word in words]
    position = [pos for pos in range(len(words))]
    context_type = [identifier_context for _ in range(len(words))]
    normalized_pos = [0 if i == 0 else (2 if i == len(words) - 1 else 1) for i in range(len(words))]
    
    word_data = pd.DataFrame(words, columns=['WORD'])
    pos_data = pd.DataFrame(max_pos, columns=['MAXPOSITION'])
    normalized_data = pd.DataFrame(normalized_pos, columns=['NORMALIZED_POSITION'])
    position_data = pd.DataFrame(position, columns=['POSITION'])
    context_data = pd.DataFrame(context_type, columns=['CONTEXT'])
    
    context_data['CONTEXT'] = context_data['CONTEXT'].apply(context_to_number)
    data = pd.concat([word_data, pos_data, normalized_data, context_data, position_data], axis=1)
    
    data = createVerbVectorFeature(data, app.model_data.ModelGensimEnglish)
    data = createDeterminerVectorFeature(data, app.model_data.ModelGensimEnglish)
    data = createConjunctionVectorFeature(data, app.model_data.ModelGensimEnglish)
    data = createPrepositionVectorFeature(data, app.model_data.ModelGensimEnglish)
    data = createPreambleVectorFeature("CODE", data, app.model_data.ModelTokens)
    data = createPreambleVectorFeature("METHOD", data, app.model_data.ModelMethods)
    data = createPreambleVectorFeature("ENGLISH", data, app.model_data.ModelGensimEnglish)
    data = createLetterFeature(data)
    data = wordPosTag(data)
    
    data['NLTK_POS'] = data['NLTK_POS'].astype('category')
    data['NLTK_POS'] = data['NLTK_POS'].cat.codes

    data = createSimilarityToVerbFeature("METHODV", app.model_data.ModelMethods, data)
    data = createSimilarityToVerbFeature("ENGLISHV", app.model_data.ModelGensimEnglish, data)
    data = createSimilarityToNounFeature("METHODN", app.model_data.ModelMethods, data)
    data = createSimilarityToNounFeature("ENGLISHN", app.model_data.ModelGensimEnglish, data)
    data = createDeterminerFeature(data)
    data = createDigitFeature(data)
    data = createPrepositionFeature(data)

    input_model = 'output/model_RandomForestClassifier.pkl'
    clf = joblib.load(input_model)

    return list(annotate_identifier(clf, data))
    
def context_to_number(context):
    if context == "ATTRIBUTE":
        return 1
    elif context == "CLASS":
        return 2
    elif context == "DECLARATION":
        return 3
    elif context == "FUNCTION":
        return 4 
    elif context == "PARAMETER":
        return 5

def annotate_identifier(clf, data):
    independent_variables_add = ['NORMALIZED_POSITION', 'LAST_LETTER', 'CONTEXT', 'MAXPOSITION', 'NLTK_POS', 'POSITION', 'VERB_SCORE', 'DET_SCORE', 'PREP_SCORE', 'CONJ_SCORE', 'PREPOSITION', 'DETERMINER', 'ENGLISHV_SCORE', 'ENGLISHN_SCORE','METHODN_SCORE', 'METHODV_SCORE', 'CODEPRE_SCORE', 'METHODPRE_SCORE', 'ENGLISHPRE_SCORE']
    df_features = pd.DataFrame(data, columns=independent_variables_add)
    y_pred = clf.predict(df_features)
    return y_pred