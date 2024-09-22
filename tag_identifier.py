import os, joblib
import pandas as pd
from feature_generator import *
from flask import Flask
from spiral import ronin
from create_models import createModel
app = Flask(__name__)

class ModelData:
    def __init__(self, wordCount, modelTokens, modelMethods, modelGensimEnglish) -> None:
        """
        Initialize an instance of the ModelData class with word vector models.

        Args:
            ModelTokens: Word vectors model for tokens.
            ModelMethods: Word vectors model for methods.
            ModelGensimEnglish: Word vectors model for general English words.
        """
        self.wordCount = wordCount
        self.modelTokens = modelTokens
        self.modelMethods = modelMethods
        self.modelGensimEnglish = modelGensimEnglish

def initialize_model():
    """
    Initialize and load word vectors for the application.

    This function initializes and loads word vectors using the 'createModel' function and stores them in the
    'app.model_data' attribute for later use.

    Returns:
        None
    """
    print("Loading word vectors!!")
    wordCount, modelTokens, modelMethods, modelGensimEnglish = createModel()
    print("Word vectors loaded!!")

    app.model_data = ModelData(wordCount, modelTokens, modelMethods, modelGensimEnglish)

def start_server():
    """
    Initialize the model and start the server.

    This function first initializes the model by calling the 'initialize_model' function. Then, it starts the server using
    the Flask 'app.run' method, allowing incoming HTTP requests to be handled.

    Returns:
        None
    """
    initialize_model()
    print("Starting server!!")
    app.run(host='0.0.0.0')


@app.route('/<identifier_name>/<identifier_context>')
def listen(identifier_name: str, identifier_context: str) -> List:
    """
    Process a web request to analyze an identifier within a specific context.

    This route function takes two URL parameters (identifier_name, and identifier_context) from an
    incoming HTTP request and performs data preprocessing and feature extraction on the identifier_name.
    It then uses a trained classifier to annotate the identifier with part-of-speech tags and other linguistic features.

    Args:
        identifier_name (str): The name of the identifier to be analyzed.
        identifier_context (str): The context in which the identifier appears.

    Returns:
        List: A list of annotations for the identifier, including part-of-speech tags and other linguistic features.
    """
    print(f"INPUT: {identifier_name} {identifier_context}")
   
    words = ronin.split(identifier_name)
    data = pd.DataFrame({
        'WORD': words,
        'IDENTIFIER': identifier_name,
        'SPLIT_IDENTIFIER': ' '.join(words),
        'MAXPOSITION': [len(words)] * len(words),
        'NORMALIZED_POSITION': [0 if i == 0 else (2 if i == len(words) - 1 else 1) for i in range(len(words))],
        'POSITION': range(len(words)),
        'CONTEXT_NUMBER': identifier_context,
        'LANGUAGE': 'C++'
    })
    
    data['CONTEXT_NUMBER'] = data['CONTEXT_NUMBER'].apply(context_to_number)

    feature_list = ['LAST_LETTER', 'NLTK_POS', 'VERB_SCORE', 'DET_SCORE', 'PREP_SCORE',
                    'CONJ_SCORE', 'PREPOSITION', 'DETERMINER', 'ENGLISHV_SCORE', 'CONTAINSLISTVERB',
                    'ENGLISHN_SCORE', 'METHODN_SCORE', 'METHODV_SCORE', 'CODEPRE_SCORE', 'WORD_COUNT', 'LANGUAGE',
                    'METHODPRE_SCORE', 'ENGLISHPRE_SCORE', 'CONTAINSDIGIT', 'CONTAINSCLOSEDSET', 'SECOND_LAST_LETTER']


    data = createFeatures(data, feature_list, app.model_data.wordCount, app.model_data.modelTokens, app.model_data.modelMethods, app.model_data.modelGensimEnglish)

    # Convert categorical variables to numeric
    categorical_features = ['NLTK_POS', 'LANGUAGE']
    for feature in categorical_features:
        if feature in data.columns:
            data[feature] = data[feature].astype('category').cat.codes

    # Load and apply the classifier
    clf = joblib.load('output/model_GradientBoostingClassifier.pkl')
    return list(annotate_identifier(clf, data))
    
def context_to_number(context):
    """
    Convert a textual context description to a numerical representation.

    This function takes a context description as a string and maps it to a numerical representation according to a
    predefined mapping.

    Args:
        context (str): The textual context description.

    Returns:
        int: The numerical representation of the context.

    Raises:
        ValueError: If the provided context is not one of the predefined values.

    Example:
        numeric_context = context_to_number("CLASS")
    """
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
    """
    Annotate identifier tokens using a trained classifier.

    This function takes a trained classifier and a dataset containing features for identifier tokens. It applies the
    classifier to predict labels for the identifier tokens.

    Args:
        clf (Classifier): The trained classifier model.
        data (pd.DataFrame): A DataFrame containing features for identifier tokens. The columns of the DataFrame should
                             match the feature names used during training.

    Returns:
        np.array: An array of predicted labels for the identifier tokens.

    Example:
        clf = load_classifier('trained_classifier.pkl')
        data = pd.DataFrame(...)  # Create a DataFrame with feature data.
        predictions = annotate_identifier(clf, data)
    """
    
    independent_variables_add = ['NORMALIZED_POSITION', 'LAST_LETTER', 'CONTEXT_NUMBER', 'MAXPOSITION',
                                      'NLTK_POS', 'POSITION', 'VERB_SCORE', 'DET_SCORE', 'PREP_SCORE',
                                      'CONJ_SCORE', 'PREPOSITION', 'DETERMINER', 'ENGLISHV_SCORE', 'CONTAINSLISTVERB',
                                      'ENGLISHN_SCORE', 'METHODN_SCORE', 'METHODV_SCORE', 'CODEPRE_SCORE', 'WORD_COUNT', 'LANGUAGE',
                                      'METHODPRE_SCORE', 'ENGLISHPRE_SCORE', 'CONTAINSDIGIT', 'CONTAINSCLOSEDSET', 'SECOND_LAST_LETTER']
    
    df_features = pd.DataFrame(data, columns=independent_variables_add)
    y_pred = clf.predict(df_features)
    return y_pred