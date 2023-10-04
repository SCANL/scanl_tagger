import logging, os, joblib
import pandas as pd
from feature_generator import *
root_logger = logging.getLogger(__name__)
from flask import Flask
from spiral import ronin
app = Flask(__name__)

print("Loading word vectors!!")
modelTokens, modelMethods, modelGensimEnglish = createModel()
print("Word vectors loaded!!")

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
    context_type = [identifier_context for _ in range(len(words))]
    normalized_pos = [0 if i == 0 else (2 if i == len(words) - 1 else 1) for i in range(len(words))]
    word_data = pd.DataFrame(words, columns=['WORD'])
    pos_data = pd.DataFrame(max_pos, columns=['MAXPOSITION'])
    normalized_data = pd.DataFrame(normalized_pos, columns=['NORMALIZED_POSITION'])
    context_data = pd.DataFrame(context_type, columns=['CONTEXT'])
    
    context_data['CONTEXT'] = context_data['CONTEXT'].apply(context_to_number)
    data = pd.concat([word_data, pos_data, normalized_data, context_data], axis=1)
    
    data = createVerbVectorFeature(data, modelGensimEnglish)
    data = createDeterminerVectorFeature(data, modelGensimEnglish)
    data = createConjunctionVectorFeature(data, modelGensimEnglish)
    data = createPrepositionVectorFeature(data, modelGensimEnglish)
    data = createPreambleVectorFeature("CODE", data, modelTokens)
    data = createPreambleVectorFeature("METHOD", data, modelMethods)
    data = createPreambleVectorFeature("ENGLISH", data, modelGensimEnglish)
    data = createLetterFeature(data)
    data = wordPosTag(data)
    
    data['NLTK_POS'] = data['NLTK_POS'].astype('category')
    data['NLTK_POS'] = data['NLTK_POS'].cat.codes

    data = createSimilarityToVerbFeature("METHODV", modelMethods, data)
    data = createSimilarityToVerbFeature("ENGLISHV", modelGensimEnglish, data)
    data = createSimilarityToNounFeature("METHODN", modelMethods, data)
    data = createSimilarityToNounFeature("ENGLISHN", modelGensimEnglish, data)
    data = createDeterminerFeature(data)
    data = createDigitFeature(data)
    data = createPrepositionFeature(data)

    input_model = 'output/model_RandomForestClassifier.pkl'
    clf = joblib.load(input_model)

    annotate_word(clf, data)
    

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

def annotate_word(clf, data):
    independent_variables_add = ['NORMALIZED_POSITION', 'LAST_LETTER', 'CONTEXT', 'MAXPOSITION', 'NLTK_POS', 'POSITION', 'VERB_SCORE', 'DET_SCORE', 'PREP_SCORE', 'CONJ_SCORE', 'PREPOSITION', 'DETERMINER', 'ENGLISHV_SCORE', 'ENGLISHN_SCORE','METHODN_SCORE', 'METHODV_SCORE', 'CODEPRE_SCORE', 'METHODPRE_SCORE', 'ENGLISHPRE_SCORE']
    df_features = pd.DataFrame(data, columns=independent_variables_add)
    print(data)
    y_pred = clf.predict(df_features)
    return y_pred[0]

if __name__ == '__main__':
    app.run(host='0.0.0.0')