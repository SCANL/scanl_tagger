import os, joblib
import pandas as pd
from feature_generator import *
from flask import Flask
from spiral import ronin
from create_models import createModel, stable_features, mutable_feature_list
app = Flask(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
class ModelData:
    def __init__(self, modelTokens, modelMethods, modelGensimEnglish, wordCount) -> None:
        """
        Initialize an instance of the ModelData class with word vector models.

        Args:
            ModelTokens: Word vectors model for tokens.
            ModelMethods: Word vectors model for methods.
            ModelGensimEnglish: Word vectors model for general English words.
        """
        self.modelTokens = modelTokens
        self.modelMethods = modelMethods
        self.modelGensimEnglish = modelGensimEnglish
        self.wordCount = wordCount

def initialize_model():
    """
    Initialize and load word vectors for the application, and load a word count DataFrame.

    This function initializes and loads word vectors using the 'createModel' function, and loads word counts 
    from a JSON file into a Pandas DataFrame for use in the application.

    Returns:
        tuple: (ModelData, WORD_COUNT DataFrame)
    """
    print("Loading word vectors!!")
    modelTokens, modelMethods, modelGensimEnglish = createModel(rootDir=SCRIPT_DIR)
    print("Word vectors loaded!!")
    
    # Load the word count JSON file into a DataFrame
    word_count_path = os.path.join("input", "word_count.json")
    if os.path.exists(word_count_path):
        print(f"Loading word count data from {word_count_path}...")
        word_count_df = pd.read_json(word_count_path, orient='index', typ='series').reset_index()
        word_count_df.columns = ['word', 'log_frequency']
        print("Word count data loaded!")
    else:
        print(f"Word count file not found at {word_count_path}. Initializing empty DataFrame.")
        word_count_df = pd.DataFrame(columns=['word', 'log_frequency'])
    
    # Create and store model data
    app.model_data = ModelData(modelTokens, modelMethods, modelGensimEnglish, word_count_df)

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
def listen(identifier_name: str, identifier_context: str) -> List[dict]:
    """
    Process a web request to analyze an identifier within a specific context.

    This route function takes two URL parameters (identifier_name, and identifier_context) from an
    incoming HTTP request and performs data preprocessing and feature extraction on the identifier_name.
    It then uses a trained classifier to annotate the identifier with part-of-speech tags and other linguistic features.

    Args:
        identifier_name (str): The name of the identifier to be analyzed.
        identifier_context (str): The context in which the identifier appears.

    Returns:
        List[dict]: A list of dictionaries containing words and their predicted POS tags.
    """
    print(f"INPUT: {identifier_name} {identifier_context}")
   
    # Split identifier_name into words
    words = identifier_name.split('_')
    
    # Create initial data frame
    data = pd.DataFrame({
        'WORD': words,
        'SPLIT_IDENTIFIER': ' '.join(words),
        'CONTEXT_NUMBER': context_to_number(identifier_context),  # Predefined context number
    })

    # # Add word count column using app.model_data.wordCount
    # if hasattr(app.model_data, 'wordCount') and not app.model_data.wordCount.empty:
    #     word_count_dict = app.model_data.wordCount.set_index('word')['log_frequency'].to_dict()
    #     data['WORD_COUNT'] = data['WORD'].str.lower().map(word_count_dict).fillna(0)
    # else:
    #     print("Word count data is missing or empty; setting WORD_COUNT to 0.")

    # Add features to the data
    data = createFeatures(
        data, 
        mutable_feature_list,
        modelGensimEnglish=app.model_data.modelGensimEnglish,
    )
    
    categorical_features = ['NLTK_POS']
    category_variables = []

    for category_column in categorical_features:
        if category_column in data.columns:
            category_variables.append(category_column)
            data.loc[:, category_column] = data[category_column].astype(str)

    for category_column in category_variables:
        # Explicitly handle categorical conversion
        unique_values = data[category_column].unique()
        category_map = {}
        for value in unique_values:
            if value in universal_to_custom:
                category_map[value] = custom_to_numeric[universal_to_custom[value]]
            else:
                category_map[value] = custom_to_numeric['NM']  # Assign 'NM' (8) for unknown categories

        data.loc[:, category_column] = data[category_column].map(category_map)
    # app.model_data.modelTokens, 
    # app.model_data.modelMethods, 
    # app.model_data.modelGensimEnglish
    # Convert categorical variables to numeric
    # Load and apply the classifier
    clf = joblib.load(os.path.join(SCRIPT_DIR, 'output', 'model_GradientBoostingClassifier.pkl'))
    predicted_tags = annotate_identifier(clf, data)

    # Combine words and their POS tags into a parseable format
    result = [{'word': word, 'pos_tag': tag} for word, tag in zip(words, predicted_tags)]

    return result
    
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
    """
    # Drop unnecessary columns
    data = data.drop(columns=['WORD', 'SPLIT_IDENTIFIER'], errors='ignore')

    # Ensure only the features used during training are included
    trained_features = clf.feature_names_in_  # Features expected by the classifier
    missing_features = set(trained_features) - set(data.columns)
    extra_features = set(data.columns) - set(trained_features)

    if missing_features:
        raise ValueError(f"The following expected features are missing: {missing_features}")
    if extra_features:
        print(f"Warning: The following unused features are being ignored: {extra_features}")
        data = data[trained_features]

    # Ensure feature order matches the trained model
    df_features = data[trained_features]
    
    print("THESE")
    print(df_features)
    
    print("THOSE")
    print(clf.feature_names_in_)

    # Make predictions
    y_pred = clf.predict(df_features)
    return y_pred