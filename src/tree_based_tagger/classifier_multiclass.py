import json, os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from src.tree_based_tagger.feature_generator import custom_to_numeric, universal_to_custom, createFeatures
from src.tree_based_tagger.create_models import createModel, stable_features, mutable_feature_list, columns_to_drop
import pandas as pd
from enum import Enum
import multiprocessing
import os, sqlite3, random
import pandas as pd
import numpy as np
from datetime import datetime

class TrainingAlgorithm(Enum):
    RANDOM_FOREST = "RandomForest"
    DECISION_TREE = "DecisionTree"
    XGBOOST = "XGBoost"
    LOGISTIC = "Logistic"
    SVC = "SVC"
    LINEAR_SVC = "LinearSVC"
    ADA_BOOST = "AdaBoost"
    ADA_BOOST_FOREST = "AdaBoostForest"
    BAGGING = "Bagging"
    GRADIENT_BOOSTING = "GradientBoosting"
    MULTI_NB = "MultiNB"
    BERNOULLI = "Bernoulli"
    K_NEIGHBORS = "KNeighbors"
    HISTOGRAMBOOST = "Histogram"


class TrainTestvalidationData:
    """
    A class to encapsulate data for classification analysis.

    Attributes:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target labels.
        X_train (pd.DataFrame): The training feature data.
        X_validation (pd.DataFrame): The validation feature data.
        X_test (pd.DataFrame): The testing feature data.
        y_train (pd.Series): The training target labels.
        y_validation (pd.Series): The validation target labels.
        y_test (pd.Series): The testing target labels.
        X_train_original (pd.DataFrame): The original training feature data.
        X_test_original (pd.DataFrame): The original testing feature data.
        labels (array-like): An array of unique labels in the target.

    Methods:
        No specific methods are defined in this class.
    """
    def __init__(self, X_train, X_test, y_train, y_test, X_train_original, X_test_original, labels):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_original = X_train_original
        self.X_test_original = X_test_original
        self.labels = labels

def load_config_tree(SCRIPT_DIR):
    # Mimic Python-based config instead of JSON
    config = {
        'script_dir': SCRIPT_DIR,
        'input_file': os.path.join(SCRIPT_DIR, 'input', 'scanl_tagger_training_db_11_29_2024.db'),
        'sql_statement': 'select * from training_set',
        'identifier_column': "ID",
        'dependent_variable': 'CORRECT_TAG',
        'pyrandom_seed': random.randint(0, 2**32 - 1),
        'trainingSeed': random.randint(0, 2**32 - 1),
        'classifierSeed': random.randint(0, 2**32 - 1),
        'npseed': random.randint(0, 2**32 - 1),
        'independent_variables': stable_features + mutable_feature_list
    }
    print(config)
    return config

def read_input(sql, features, conn, config):
    """
    Read input data from an SQLite database and preprocess it.

    This function reads data from the specified SQL query and database connection, shuffles the rows, and then applies
    a preprocessing function called 'createFeatures' to create additional features.

    Args:
        sql (str): The SQL query to fetch data from the database.
        conn (sqlite3.Connection): The SQLite database connection.

    Returns:
        pandas.DataFrame: A DataFrame containing the preprocessed input data.
    """
    input_data = pd.read_sql_query(sql, conn)
    print(" --  --  --  -- Read " + str(len(input_data)) + " input rows --  --  --  -- ")
    print(input_data.columns)
    input_data_copy = input_data.copy()
    rows = input_data_copy.values.tolist()
    random.shuffle(rows)
    shuffled_input_data = pd.DataFrame(rows, columns=input_data.columns)
    modelTokens, modelMethods, modelGensimEnglish = createModel(rootDir=config['script_dir'])
    input_data = createFeatures(shuffled_input_data, features, modelGensimEnglish=modelGensimEnglish, modelTokens=modelTokens, modelMethods=modelMethods)
    return input_data

def train_tree(config):
    """
    Train a part of speech tagger model using specified features and a training dataset.
    This function reads data from an SQLite database, preprocesses it, and performs classification using a specified set
    of features. The results are written to an output file, including information about the training process and the
    distribution of labels in the training data.
    Args:
        config (dict): A dictionary containing configuration data.
    Returns:
        None
    """
   
    # Extract configuration values from the 'config' dictionary
    input_file = config['input_file']
    sql_statement = config['sql_statement']
    identifier_column = config['identifier_column']
    dependent_variable = config['dependent_variable']
    pyrandom_seed = config['pyrandom_seed']
    trainingSeed = config['trainingSeed']
    classifierSeed = config['classifierSeed']
   
    np.random.seed(config['npseed'])
    random.seed(pyrandom_seed)
    independent_variables = config['independent_variables']
    
    # ###############################################################
    print(" --  -- Started: Reading Database --  -- ")
    connection = sqlite3.connect(input_file)
    df_input = read_input(sql_statement, independent_variables, connection, config)
    print(" --  -- Completed: Reading Input --  -- ")
    # ###############################################################
    
    # Create an explicit copy to avoid SettingWithCopyWarning
    #independent_variables.remove("EMB_FEATURES")
    df_features = df_input[independent_variables].copy()
    df_class = df_input[[dependent_variable]].copy()
    
    category_variables = []
    categorical_columns = ['NLTK_POS', 'PREV_POS', 'NEXT_POS']
    
    # Safely handle categorical variables
    for category_column in categorical_columns:
        if category_column in df_features.columns:
            category_variables.append(category_column)
            df_features.loc[:, category_column] = df_features[category_column].astype(str)
    
    # Ensure output directories exist
    output_dir = os.path.join(config['script_dir'], 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, 'results.txt')
    mode = 'a' if os.path.exists(filename) else 'w'
   
    with open(filename, mode) as results_text_file:
        results_text_file.write(datetime.now().strftime("%H:%M:%S") + "\n")
       
        # Print config in a readable fashion
        results_text_file.write("Configuration:\n")
        for key, value in config.items():
            results_text_file.write(f"{key}: {value}\n")
        results_text_file.write("\n")

        for category_column in category_variables:
            # Explicitly handle categorical conversion
            unique_values = df_features[category_column].unique()
            category_map = {}
            for value in unique_values:
                print(value)
                if value in universal_to_custom:
                    category_map[value] = custom_to_numeric[universal_to_custom[value]]
                else:
                    category_map[value] = custom_to_numeric['NOUN']  # Assign 'NM' (8) for unknown categories

            df_features.loc[:, category_column] = df_features[category_column].map(category_map)
       
        print(" --  -- Distribution of labels in corpus --  -- ")
        print(df_class[dependent_variable].value_counts())
        results_text_file.write(f"SQL: {sql_statement}\n")
        results_text_file.write(f"Features: {df_features}\n")
       
        algorithms = [TrainingAlgorithm.XGBOOST]
        #pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', None)  # Prevent line wrapping
        pd.set_option('display.max_colwidth', None)  # Show full content of each cell

        print(df_features)
        perform_classification(df_features, df_class, results_text_file,
                                                    output_dir, algorithms, trainingSeed,
                                                    classifierSeed, columns_to_drop)
def build_datasets(X, y, output_directory, trainingSeed):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Split the data into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=trainingSeed, stratify=y
    )

    # Store original copies before feature creation
    X_train_original = X_train.copy(deep=True)
    X_test_original = X_test.copy(deep=True)

    # Filter X_test to only include specific columns
    validation_columns = ['SPLIT_IDENTIFIER', 'CONTEXT_NUMBER', 'WORD']
    if not all(col in X_test.columns for col in validation_columns):
        raise ValueError(f"X_test must contain the columns {validation_columns}")

    X_test_filtered = X_test[validation_columns]
    
    # Output filtered X_test to a CSV file
    X_test_path = os.path.join(output_directory, 'X_test.csv')
    X_test_filtered.to_csv(X_test_path, index=False)
    print(f"Filtered X_test saved to: {X_test_path}")

    # Output y_test to a CSV file
    y_test_path = os.path.join(output_directory, 'y_test.csv')
    pd.DataFrame(y_test.values.ravel(), columns=['label']).to_csv(y_test_path, index=False)
    print(f"y_test saved to: {y_test_path}")
    
    # Print distribution of labels in all sets
    for name, labels in [("Training", y_train), ("Test", y_test)]:
        unique, counts = np.unique(labels, return_counts=True)
        print(f" -- Distribution of labels in {name} set -- ")
        print(dict(zip(unique, counts)))
        print()

    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel(), X_train_original, X_test_original


def perform_classification(X, y, results_text_file, output_directory, TrainingAlgorithms, trainingSeed, classifierSeed, columns_to_drop):
    """
    Perform classification using specified TrainingAlgorithms and report results.

    This function performs classification on the input data using the specified machine learning TrainingAlgorithms and reports
    various evaluation metrics to the results text file.

    Args:
        X (pandas.DataFrame): The feature data (X).
        y (numpy.ndarray): The label data (y).
        results_text_file (file): The file to write the results to.
        output_directory (str): The directory where additional output files will be saved.
        TrainingAlgorithms (list): A list of TrainingAlgorithm enum values specifying the TrainingAlgorithms to use.
        trainingSeed (int): The random seed for data splitting during training.
        classifierSeed (int): The random seed for the classifier.

    Returns:
        None
    """

    X_train, X_test, y_train, y_test, X_train_original, X_test_original = build_datasets(X, y, output_directory, trainingSeed)
    labels = np.unique(y_train, return_counts=False)
    
    algoData = TrainTestvalidationData(X_train, X_test, y_train, y_test, X_train_original, X_test_original, labels)
    
    results_text_file.write("Training Seed: %s\n" % trainingSeed)
    results_text_file.write("Classifier Seed: %s\n" % classifierSeed)

    scorers = {
        'accuracy': make_scorer(accuracy_score),  # Accuracy
        'weighted_f1': make_scorer(f1_score, average='weighted'),  # Weighted F1-score
        'balanced_accuracy': make_scorer(balanced_accuracy_score)  # Balanced Accuracy
    }

    for TrainingAlgorithm in TrainingAlgorithms:
        if TrainingAlgorithm == TrainingAlgorithm.XGBOOST:
            analyzeGradientBoost(results_text_file, output_directory, scorers, algoData, classifierSeed, trainingSeed, columns_to_drop)

def write_importances(results_text_file, feature_names, presult, metric_name):
    """
    Helper function to write permutation importances to the results file.
    """
    results_text_file.write(f"{metric_name} importances\n")
    for feature, value in zip(feature_names, presult.importances):
        results_text_file.write(f"{feature},{','.join(map(str, value))}\n")
    results_text_file.write("\n")
    results_text_file.write(f"mean {metric_name} importances\n")
    for feature, value in zip(feature_names, presult.importances_mean):
        results_text_file.write(f"{feature},{value}\n")
    results_text_file.write("\n")

def analyzeGradientBoost(results_text_file, output_directory, scorersKey, algoData, classifierSeed, trainingSeed, columns_to_drop):
    """
    Analyze a GradientBoostingClassifier for classification and report results.

    Args:
        results_text_file (file): The file to write the results to.
        output_directory (str): The directory where additional output files will be saved.
        scorersKey (dict): A dictionary of scoring functions.
        algoData (TrainTestvalidationData): An object containing data for analysis.
        classifierSeed (int): The random seed for the classifier.
        trainingSeed (int): The random seed for data splitting during training.
    """
    # Hyperparameter grid for Gradient Boosting
    param_gradientboost = {
        'n_estimators': [100, 200, 250],
        'learning_rate': [0.05, 0.1],
        'max_depth': [7, 8, 9],
        'subsample': [0.9, 1.0],
        'max_features': ['sqrt'],
    }

    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=trainingSeed)

    try:
        results_text_file.write("\n---------------------------GradientBoostingClassifier---------------------------\n")
        print("GradientBoostingClassifier")

        # Drop SPLIT_IDENTIFIER and WORD columns from X_train

        X_train_dropped = algoData.X_train.drop(columns=columns_to_drop, errors='ignore')
        
        max_threads = max(1, multiprocessing.cpu_count() - 3)
        
        # Grid search with cross-validation
        clf = GridSearchCV(
            GradientBoostingClassifier(random_state=classifierSeed),
            param_gradientboost,
            cv=stratified_kfold,
            scoring=scorersKey,
            n_jobs=max_threads,
            refit='weighted_f1'
        )
        clf.fit(X_train_dropped, algoData.y_train)

        # Save the trained model
        model_path = os.path.join(output_directory, "model_GradientBoostingClassifier.pkl")
        joblib.dump(clf.best_estimator_, model_path)

        # Log the best parameters
        results_text_file.write("Best parameters set found on development set:\n")
        results_text_file.write(json.dumps(clf.best_params_) + "\n")

        # Save cross-validation results to CSV
        cv_results_path = os.path.join(output_directory, "cv_results_GradientBoostingClassifier.csv")
        pd.DataFrame(clf.cv_results_).to_csv(cv_results_path)

        # Calculate permutation importances for various metrics
        best_model = clf.best_estimator_
        metrics = ['f1_weighted', 'balanced_accuracy', 'accuracy']

        for metric in metrics:
            presult = permutation_importance(
                best_model, X_train_dropped, algoData.y_train, scoring=metric, n_jobs=-1
            )
            write_importances(results_text_file, X_train_dropped.columns, presult, metric)

        # Test set predictions
        X_test_dropped = algoData.X_test.drop(columns=columns_to_drop, errors='ignore')
        y_test_pred = best_model.predict(X_test_dropped)
        test_accuracy = accuracy_score(algoData.y_test, y_test_pred)
        results_text_file.write(f"Test Accuracy: {test_accuracy:.4f}\n")

        # Save test results to CSV
        test_results_path = os.path.join(output_directory, "test_results_GradientBoostingClassifier.csv")
        test_results = pd.DataFrame({
            'Actual_label': algoData.y_test,
            'Predicted_Label': y_test_pred,
            'Accuracy': [test_accuracy] * len(algoData.y_test),
            'Word': algoData.X_test['WORD']
        })
        test_results.to_csv(test_results_path, index=False)

        # Cross-validation predictions
        y_pred_cv = cross_val_predict(best_model, X_train_dropped, algoData.y_train, cv=stratified_kfold)

        # Log cross-validation results
        results_text_file.write("--- Cross-Validation Classification Report ---\n")
        results_text_file.write(classification_report(algoData.y_train, y_pred_cv))
        results_text_file.write("\n")
        results_text_file.write(f"balanced_accuracy_score (CV): {balanced_accuracy_score(algoData.y_train, y_pred_cv):.4f}\n")
        results_text_file.write(f"f1_score (macro, CV): {f1_score(algoData.y_train, y_pred_cv, average='macro'):.4f}\n")
        results_text_file.write(f"f1_score (micro, CV): {f1_score(algoData.y_train, y_pred_cv, average='micro'):.4f}\n")
        results_text_file.write(f"f1_score (weighted, CV): {f1_score(algoData.y_train, y_pred_cv, average='weighted'):.4f}\n")
        results_text_file.write(f"matthews_corrcoef (CV): {matthews_corrcoef(algoData.y_train, y_pred_cv):.4f}\n")

        # Confusion matrix
        cm_test = confusion_matrix(algoData.y_test, y_test_pred, labels=algoData.labels)
        results_text_file.write("\nConfusion Matrix (Test Set):\n")
        results_text_file.write("\n".join(",".join(map(str, row)) for row in cm_test))
        results_text_file.write("\n")

        results_text_file.write("\n------------------------------------------------------\n")
        results_text_file.flush()

    except Exception as e:
        results_text_file.write(f"Error during GradientBoostingClassifier analysis: {e}\n")
        print(f"Error: {e}")
        raise
