import logging, os, joblib
import pandas as pd
from feature_generator import *
root_logger = logging.getLogger(__name__)
from flask import Flask, g
from spiral import ronin
app = Flask(__name__)

class ModelData:
    def __init__(self, ModelTokens, ModelMethods,ModelGensimEnglish) -> None:
        self.ModelTokens = ModelTokens
        self.ModelMethods = ModelMethods
        self.ModelGensimEnglish = ModelGensimEnglish

md = None
def start_server():
    print("Loading word vectors!!")
    ModelTokens, ModelMethods, ModelGensimEnglish = createModel()
    print("Word vectors loaded!!")

    print("Starting server!!")
    app.run(host='0.0.0.0')
    print("server started!!")

    global md 
    md = ModelData(ModelTokens, ModelMethods, ModelGensimEnglish)

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

@app.route('/<identifier_type>/<identifier_name>/<identifier_context>')
def listen(identifier_type, identifier_name, identifier_context):
    root_logger.info("INPUT: {ident_type} {ident_name} {ident_context}".format(ident_type=identifier_type, ident_name=identifier_name, ident_context=identifier_context))
    
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
    
    data = createVerbVectorFeature(data, md.ModelGensimEnglish)
    data = createDeterminerVectorFeature(data, md.ModelGensimEnglish)
    data = createConjunctionVectorFeature(data, md.ModelGensimEnglish)
    data = createPrepositionVectorFeature(data, md.ModelGensimEnglish)
    data = createPreambleVectorFeature("CODE", data, md.ModelTokens)
    data = createPreambleVectorFeature("METHOD", data, md.ModelMethods)
    data = createPreambleVectorFeature("ENGLISH", data, md.ModelGensimEnglish)
    data = createLetterFeature(data)
    data = wordPosTag(data)
    
    data['NLTK_POS'] = data['NLTK_POS'].astype('category')
    data['NLTK_POS'] = data['NLTK_POS'].cat.codes

    data = createSimilarityToVerbFeature("METHODV", md.ModelMethods, data)
    data = createSimilarityToVerbFeature("ENGLISHV", md.ModelGensimEnglish, data)
    data = createSimilarityToNounFeature("METHODN", md.ModelMethods, data)
    data = createSimilarityToNounFeature("ENGLISHN", md.ModelGensimEnglish, data)
    data = createDeterminerFeature(data)
    data = createDigitFeature(data)
    data = createPrepositionFeature(data)

    input_model = 'output/model_RandomForestClassifier.pkl'
    clf = joblib.load(input_model)

    return annotate_identifier(clf, data)
    

class MSG_COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def annotate_identifier(clf, data):
    independent_variables_add = ['NORMALIZED_POSITION', 'LAST_LETTER', 'CONTEXT', 'MAXPOSITION', 'NLTK_POS', 'POSITION', 'VERB_SCORE', 'DET_SCORE', 'PREP_SCORE', 'CONJ_SCORE', 'PREPOSITION', 'DETERMINER', 'ENGLISHV_SCORE', 'ENGLISHN_SCORE','METHODN_SCORE', 'METHODV_SCORE', 'CODEPRE_SCORE', 'METHODPRE_SCORE', 'ENGLISHPRE_SCORE']
    df_features = pd.DataFrame(data, columns=independent_variables_add)
    y_pred = clf.predict(df_features)
    return y_pred