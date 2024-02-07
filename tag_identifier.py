import os, joblib
import pandas as pd
from feature_generator import *
from flask import Flask
from spiral import ronin

app = Flask(__name__)

class ModelData:
    def __init__(self, ModelTokens, ModelMethods, ModelGensimEnglish) -> None:
        """
        Initialize an instance of the ModelData class with word vector models.

        Args:
            ModelTokens: Word vectors model for tokens.
            ModelMethods: Word vectors model for methods.
            ModelGensimEnglish: Word vectors model for general English words.
        """
        self.ModelTokens = ModelTokens
        self.ModelMethods = ModelMethods
        self.ModelGensimEnglish = ModelGensimEnglish

def initialize_model():
    """
    Initialize and load word vectors for the application.

    This function initializes and loads word vectors using the 'createModel' function and stores them in the
    'app.model_data' attribute for later use.

    Returns:
        None
    """
    print("Loading word vectors!!")
    ModelTokens, ModelMethods, ModelGensimEnglish = createModel()
    print("Word vectors loaded!!")

    app.model_data = ModelData(ModelTokens, ModelMethods, ModelGensimEnglish)

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
def listen(identifier_name, identifier_context):
    """
    Process a web request to analyze an identifier within a specific context.

    This route function takes two URL parameters (identifier_name, and identifier_context) from an
    incoming HTTP request and performs several data preprocessing and feature extraction steps on the identifier_name.
    It then uses a trained classifier to annotate the identifier with part-of-speech tags and other linguistic features.

    Args:
        identifier_name (str): The name of the identifier to be analyzed.
        identifier_context (str): The context in which the identifier appears.

    Returns:
        list: A list of annotations for the identifier, including part-of-speech tags and other linguistic features.
    """
    print("INPUT: {ident_name} {ident_context}".format(ident_name=identifier_name, ident_context=identifier_context))
    
    data = pd.DataFrame({'WORD': ronin.split(identifier_name), 'IDENTIFIER':identifier_name})
    
    words = ronin.split(identifier_name)
    max_pos = [len(words) for word in words]
    position = [pos for pos in range(len(words))]
    context_type = [identifier_context for _ in range(len(words))]
    normalized_pos = [0 if i == 0 else (2 if i == len(words) - 1 else 1) for i in range(len(words))]

    pos_data = pd.DataFrame(max_pos, columns=['MAXPOSITION'])
    normalized_data = pd.DataFrame(normalized_pos, columns=['NORMALIZED_POSITION'])
    position_data = pd.DataFrame(position, columns=['POSITION'])
    context_data = pd.DataFrame(context_type, columns=['CONTEXT'])
    
    context_data['CONTEXT'] = context_data['CONTEXT'].apply(context_to_number)
    data = pd.concat([data, pos_data, normalized_data, context_data, position_data], axis=1)
    
    data = createVerbFeature(data)
    data = createIdentifierDigitFeature(data)
    data = createIdentifierClosedSetFeature(data)
    data = createIdentifierContainsVerbFeature(data)

    data = createLetterFeature(data)
    data = wordPosTag(data)
    
    data['NLTK_POS'] = data['NLTK_POS'].astype(str)
    data['NLTK_POS'] = data['NLTK_POS'].astype('category')
    data['NLTK_POS'] = data['NLTK_POS'].cat.codes
    
    data['PREVIOUS_NLTK_POS'] = data['PREVIOUS_NLTK_POS'].astype(str)
    data['PREVIOUS_NLTK_POS'] = data['PREVIOUS_NLTK_POS'].astype('category')
    data['PREVIOUS_NLTK_POS'] = data['PREVIOUS_NLTK_POS'].cat.codes

    data = createSimilarityToVerbFeature("METHODV", app.model_data.ModelMethods, data)
    data = createSimilarityToVerbFeature("ENGLISHV", app.model_data.ModelGensimEnglish, data)
    data = createSimilarityToNounFeature("METHODN", app.model_data.ModelMethods, data)
    data = createSimilarityToNounFeature("ENGLISHN", app.model_data.ModelGensimEnglish, data)
    
    data = createDeterminerFeature(data)
    data = createDigitFeature(data)
    
    data = createDeterminerVectorFeature(data, app.model_data.ModelGensimEnglish)
    data = createConjunctionVectorFeature(data, app.model_data.ModelGensimEnglish)
    data = createPrepositionVectorFeature(data, app.model_data.ModelGensimEnglish)
    
    input_model = 'output/model_RandomForestClassifier.pkl'
    clf = joblib.load(input_model)

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
    
    independent_variables_add = ['NORMALIZED_POSITION', 'LAST_LETTER', 'CONTEXT', 'MAXPOSITION',
                                'NLTK_POS', 'POSITION', 'DETERMINER', 'ENGLISHV_SCORE',
                                'ENGLISHN_SCORE', 'METHODN_SCORE', 'METHODV_SCORE', 'DIGITS', 'CONTAINSLISTVERB',
                                'CONTAINSDIGIT', 'CONTAINSCLOSEDSET', 'CONTAINSVERB', 'DET_SCORE', 'CONJ_SCORE', 'PREP_SCORE']
    
    df_features = pd.DataFrame(data, columns=independent_variables_add)
    y_pred = clf.predict(df_features)
    return y_pred