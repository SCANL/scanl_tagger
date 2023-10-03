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


@app.route('/<identifier_type>/<identifier_name>/<identifier_context>')
def listen(identifier_type, identifier_name, identifier_context):
    root_logger.info("INPUT: {ident_type} {ident_name} {ident_context}".format(ident_type=identifier_type, ident_name=identifier_name, ident_context=identifier_context))
    words = ronin.split(identifier_name)
    print(words)
    maxpos = [len(words) for word in words]
    word_data = pd.DataFrame(words, columns=['WORD'])
    pos_data = pd.DataFrame(maxpos, columns=['MAXPOS'])
    data = pd.concat([word_data, pos_data], axis=1)
    data = createVerbVectorFeature(data, modelGensimEnglish)
    data = createDeterminerVectorFeature(data, modelGensimEnglish)
    data = createConjunctionVectorFeature(data, modelGensimEnglish)
    data = createPrepositionVectorFeature(data, modelGensimEnglish)
    data = createPreambleVectorFeature("CODE", data, modelTokens)
    data = createPreambleVectorFeature("METHOD", data, modelMethods)
    data = createPreambleVectorFeature("ENGLISH", data, modelGensimEnglish)
    data = createLetterFeature(data)
    #data = maxPosition(data)
    data = wordPosTag(data)
    data = createSimilarityToVerbFeature("METHODV", modelMethods, data)
    data = createSimilarityToVerbFeature("ENGLISHV", modelGensimEnglish, data)
    data = createSimilarityToNounFeature("METHODN", modelMethods, data)
    data = createSimilarityToNounFeature("ENGLISHN", modelGensimEnglish, data)
    data = createDeterminerFeature(data)
    data = createDigitFeature(data)
    data = createPrepositionFeature(data)
    # data = firstWordLength(data)
    # data = firstWordCaps(data)
    

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

def annotate_word(params):
    input_model = 'output/model_RandomForestClassifier.pkl'

    data = {
        'NORMALIZED_POSITION': params['normalized_length'],
        'LAST_LETTER': params['last_letter'],
        'CONTEXT': params['code_context'],
        'MAXPOSITION': params['max_position'],
        'NLTK_POS': params['nltk_pos'],
        'POSITION': params['position'],
        'VERB_SCORE': params['verb_score'],
        'DET_SCORE': params['det_score'],
        'PREP_SCORE': params['prep_score'],
        'CONJ_SCORE': params['conj_score'],
        'PREPOSITION': params['prep'],
        'DETERMINER': params['det'],
        'ENGLISHV_SCORE': params['englishv_score'],
        'ENGLISHN_SCORE': params['englishn_score'],
        'METHODN_SCORE': params['methodn_score'],
        'METHODV_SCORE': params['methodv_score'],
        'CODEPRE_SCORE': params['codepre_score'],
        'METHODPRE_SCORE': params['methodpre_score'],
        'ENGLISHPRE_SCORE': params['englishpre_score'],
        'FIRST_WORD_LENGTH': params['first_word_len'],
        'FIRST_WORD_CAPS': params['first_word_caps'],
    }

    # df_features = pd.DataFrame(data, columns=independent_variables_base + independent_variables_add[0])

    # clf = joblib.load(input_model)
    # y_pred = clf.predict(df_features)
    # return y_pred[0]

if __name__ == '__main__':
    app.run(host='0.0.0.0')