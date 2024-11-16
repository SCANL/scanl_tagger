import json

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from imblearn.metrics import classification_report_imbalanced
import print_utility_functions as utils
from enum import Enum
import random
from feature_generator import *

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
    def __init__(self, X_train, X_validation, X_test, y_train, y_validation, y_test, X_train_original, X_test_original, labels):
        self.X_train = X_train
        self.X_validation = X_validation
        self.X_test = X_test
        self.y_train = y_train
        self.y_validation = y_validation
        self.y_test = y_test
        self.X_train_original = X_train_original
        self.X_test_original = X_test_original
        self.labels = labels

def build_datasets(X, y, output_directory, trainingSeed):
    # Split the data into training (70%) and temporary (30%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, 
                                                       random_state=trainingSeed, 
                                                       stratify=y)

    # Split the temporary set into validation (15%) and testing (15%) sets
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, 
                                                                 test_size=0.50, 
                                                                 random_state=trainingSeed, 
                                                                 stratify=y_temp)
    
    # Store original copies before feature creation
    X_train_original = X_train.copy(deep=True)
    X_test_original = X_temp.copy(deep=True)
    
    if 'WORD' in X_train.columns:
        # Calculate word frequencies from training data only
        word_frequencies = calculate_word_frequencies(X_train['WORD'])
        
        # Apply the word frequencies to all sets
        X_train = apply_word_counts(X_train, word_frequencies)
        X_validation = apply_word_counts(X_validation, word_frequencies)
        X_test = apply_word_counts(X_test, word_frequencies)
    
    # Print distribution of labels in all sets
    for name, labels in [("Training", y_train), ("Validation", y_validation), ("Test", y_test)]:
        unique, counts = np.unique(labels, return_counts=True)
        print(f" -- Distribution of labels in {name} set -- ")
        print(dict(zip(unique, counts)))
        print()

    return X_train, X_validation, X_test, y_train.values.ravel(), y_validation.values.ravel(), y_test.values.ravel(), X_train_original, X_test_original

def perform_classification(X, y, results_text_file, output_directory, TrainingAlgorithms, trainingSeed, classifierSeed):
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
    X_train, X_validation, X_test, y_train, y_validation, y_test, X_train_original, X_test_original = build_datasets(X, y, output_directory, trainingSeed)
    labels = np.unique(y_train, return_counts=False)
    
    algoData = TrainTestvalidationData(X_train, X_validation, X_test, y_train, y_validation, y_test, X_train_original, X_test_original, labels)
    results_text_file.write("Training Seed: %s\n" % trainingSeed)
    results_text_file.write("Classifier Seed: %s\n" % classifierSeed)

    scorers = {
        'accuracy': make_scorer(accuracy_score),  # Accuracy
        'weighted_f1': make_scorer(f1_score, average='weighted'),  # Weighted F1-score
        'balanced_accuracy': make_scorer(balanced_accuracy_score)  # Balanced Accuracy
    }

    for TrainingAlgorithm in TrainingAlgorithms:
        if TrainingAlgorithm == TrainingAlgorithm.RANDOM_FOREST:
            analyzeRandomForest(results_text_file, output_directory, scorers, algoData, classifierSeed, trainingSeed)
        if TrainingAlgorithm == TrainingAlgorithm.DECISION_TREE:
            analyzeDecisionTree(results_text_file, output_directory, scorers, algoData, classifierSeed, trainingSeed)
        if TrainingAlgorithm == TrainingAlgorithm.XGBOOST:
            analyzeGradientBoost(results_text_file, output_directory, scorers, algoData, classifierSeed, trainingSeed)

def analyzeRandomForest(results_text_file, output_directory, scorersKey, algoData, classifierSeed, trainingSeed):
    """
    Analyze a RandomForestClassifier for classification and report results.

    This function performs analysis on a RandomForestClassifier for classification and reports various evaluation metrics
    to the results text file.

    Args:
        results_text_file (file): The file to write the results to.
        output_directory (str): The directory where additional output files will be saved.
        scorersKey (dict): A dictionary of scoring functions.
        algoData (TrainTestvalidationData): An object containing data for analysis.
        classifierSeed (int): The random seed for the classifier.
        trainingSeed (int): The random seed for data splitting during training.

    Returns:
        None
    """
    param_randomforest = {
        'n_estimators': [200, 220, 240, 260],
        'max_depth': range(1, 30),
        'criterion': ['gini', 'entropy'],
        'bootstrap': [True],
        'class_weight':["balanced", "balanced_subsample"],
        'max_features':["sqrt", "log2"],
        #'warm_start':[True]
    }
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=trainingSeed)
    results_text_file.write("\n---------------------------RandomForestClassifier---------------------------\n")
    print("RandomForestClassifier")
    clf = GridSearchCV(RandomForestClassifier(random_state=classifierSeed), param_randomforest, cv=stratified_kfold,
                       scoring=scorersKey, n_jobs=-1, refit='weighted_f1',
                       error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)

    joblib.dump(clf, '%s/model_RandomForestClassifier.pkl' % output_directory)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_RandomForestClassifier.csv" % output_directory)

    presult_f1 = permutation_importance(clf.best_estimator_, algoData.X_train, algoData.y_train, scoring='f1_weighted')
    results_text_file.write("f1_weighted importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_f1.importances):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    results_text_file.write("mean f1_weighted importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_f1.importances_mean):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=value))
    results_text_file.write("\n")

    presult_balanced = permutation_importance(clf.best_estimator_, algoData.X_train, algoData.y_train, scoring='balanced_accuracy')
    results_text_file.write("balanced_accuracy importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_balanced.importances):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    results_text_file.write("mean balanced_accuracy importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_balanced.importances_mean):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=value))
    results_text_file.write("\n")

    presult_accuracy = permutation_importance(clf.best_estimator_, algoData.X_train, algoData.y_train, scoring='accuracy')
    results_text_file.write("accuracy importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_accuracy.importances):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("\n")
    results_text_file.write("The scores are computed on the full evaluation set.")
    results_text_file.write("\n")

    # Get the best estimator
    best_model = clf.best_estimator_
    
    y_validation_pred = best_model.predict(algoData.X_validation)
    
    # Calculate accuracy
    accuracy = accuracy_score(algoData.y_validation, y_validation_pred)

    # Create DataFrame with actual and predicted labels
    validation_results = pd.DataFrame({
        'Actual_label': algoData.y_validation,
        'Predicted_Label': y_validation_pred,
        'Accuracy': [accuracy] * len(algoData.y_validation)
    })

    # Save the DataFrame to a CSV file
    validation_results.to_csv('validation_results.csv', index=False)

    # Perform cross-validation and obtain predictions
    y_pred_cv = cross_val_predict(best_model, algoData.X_train, algoData.y_train, cv=stratified_kfold)

    results_text_file.write('--- Cross-Validation Classification Report ---\n')
    results_text_file.write(classification_report(algoData.y_train, y_pred_cv))
    results_text_file.write("\n")

    results_text_file.write('balanced_accuracy_score (CV) :')
    results_text_file.write(str(balanced_accuracy_score(algoData.y_train, y_pred_cv)))
    results_text_file.write("\n")
    results_text_file.write('f1_score (macro, CV) :')
    results_text_file.write(str(f1_score(algoData.y_train, y_pred_cv, average='macro')))
    results_text_file.write("\n")
    results_text_file.write('f1_score (micro, CV) :')
    results_text_file.write(str(f1_score(algoData.y_train, y_pred_cv, average='micro')))
    results_text_file.write("\n")
    results_text_file.write('f1_score (weighted, CV) :')
    results_text_file.write(str(f1_score(algoData.y_train, y_pred_cv, average='weighted')))
    results_text_file.write("\n")
    results_text_file.write('matthews_corrcoef (CV) :')
    results_text_file.write(str(matthews_corrcoef(algoData.y_train, y_pred_cv)))
    results_text_file.write("\n")
    results_text_file.flush()
    utils.print_prediction_results(algoData.X_train.index, y_pred_cv, algoData.y_train, 'RandomForestClassifier',
                                output_directory)
    cm_cv = confusion_matrix(algoData.y_train, y_pred_cv, labels=algoData.labels)
    utils.print_cm(cm_cv, algoData.labels, classifier='RandomForestClassifier', output_directory=output_directory)
    results_text_file.write("\n------------------------------------------------------\n")

def analyzeGradientBoost(results_text_file, output_directory, scorersKey, algoData, classifierSeed, trainingSeed):
    """
    Analyze a GradientBoostingClassifier for classification and report results.

    This function performs analysis on a GradientBoostingClassifier for classification and reports various evaluation metrics
    to the results text file.

    Args:
        results_text_file (file): The file to write the results to.
        output_directory (str): The directory where additional output files will be saved.
        scorersKey (dict): A dictionary of scoring functions.
        algoData (TrainTestvalidationData): An object containing data for analysis.
        classifierSeed (int): The random seed for the classifier.
        trainingSeed (int): The random seed for data splitting during training.

    Returns:
        None
    """
    param_gradientboost = {
        'n_estimators': [100, 200, 250],
        'learning_rate': [0.1, 0.2],
        'max_depth': [5, 6],
        'subsample': [0.9, 1.0],
        'max_features': ['sqrt'],
    }
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=trainingSeed)
    results_text_file.write("\n---------------------------GradientBoostingClassifier---------------------------\n")
    print("GradientBoostingClassifier")
    clf = GridSearchCV(GradientBoostingClassifier(random_state=classifierSeed), param_gradientboost, cv=stratified_kfold,
                       scoring=scorersKey, n_jobs=-1, refit='weighted_f1',
                       error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)

    joblib.dump(clf, '%s/model_GradientBoostingClassifier.pkl' % output_directory)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_GradientBoostingClassifier.csv" % output_directory)

    presult_f1 = permutation_importance(clf.best_estimator_, algoData.X_train, algoData.y_train, scoring='f1_weighted')
    results_text_file.write("f1_weighted importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_f1.importances):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    results_text_file.write("mean f1_weighted importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_f1.importances_mean):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=value))
    results_text_file.write("\n")

    presult_balanced = permutation_importance(clf.best_estimator_, algoData.X_train, algoData.y_train, scoring='balanced_accuracy')
    results_text_file.write("balanced_accuracy importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_balanced.importances):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    results_text_file.write("mean balanced_accuracy importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_balanced.importances_mean):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=value))
    results_text_file.write("\n")

    presult_accuracy = permutation_importance(clf.best_estimator_, algoData.X_train, algoData.y_train, scoring='accuracy')
    results_text_file.write("accuracy importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_accuracy.importances):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("\n")
    results_text_file.write("The scores are computed on the full evaluation set.")
    results_text_file.write("\n")

    # Get the best estimator
    best_model = clf.best_estimator_
    
    y_validation_pred = best_model.predict(algoData.X_validation)
    
    # Calculate accuracy
    accuracy = accuracy_score(algoData.y_validation, y_validation_pred)

    # Create DataFrame with actual and predicted labels
    validation_results = pd.DataFrame({
        'Actual_label': algoData.y_validation,
        'Predicted_Label': y_validation_pred,
        'Accuracy': [accuracy] * len(algoData.y_validation)
    })

    # Save the DataFrame to a CSV file
    validation_results.to_csv('validation_results.csv', index=False)

    # Perform cross-validation and obtain predictions
    y_pred_cv = cross_val_predict(best_model, algoData.X_train, algoData.y_train, cv=stratified_kfold)

    results_text_file.write('--- Cross-Validation Classification Report ---\n')
    results_text_file.write(classification_report(algoData.y_train, y_pred_cv))
    results_text_file.write("\n")

    results_text_file.write('balanced_accuracy_score (CV) :')
    results_text_file.write(str(balanced_accuracy_score(algoData.y_train, y_pred_cv)))
    results_text_file.write("\n")
    results_text_file.write('f1_score (macro, CV) :')
    results_text_file.write(str(f1_score(algoData.y_train, y_pred_cv, average='macro')))
    results_text_file.write("\n")
    results_text_file.write('f1_score (micro, CV) :')
    results_text_file.write(str(f1_score(algoData.y_train, y_pred_cv, average='micro')))
    results_text_file.write("\n")
    results_text_file.write('f1_score (weighted, CV) :')
    results_text_file.write(str(f1_score(algoData.y_train, y_pred_cv, average='weighted')))
    results_text_file.write("\n")
    results_text_file.write('matthews_corrcoef (CV) :')
    results_text_file.write(str(matthews_corrcoef(algoData.y_train, y_pred_cv)))
    results_text_file.write("\n")
    results_text_file.flush()
    utils.print_prediction_results(algoData.X_train.index, y_pred_cv, algoData.y_train, 'GradientBoostingClassifier',
                                output_directory)
    cm_cv = confusion_matrix(algoData.y_train, y_pred_cv, labels=algoData.labels)
    utils.print_cm(cm_cv, algoData.labels, classifier='GradientBoostingClassifier', output_directory=output_directory)
    results_text_file.write("\n------------------------------------------------------\n")

def analyzeDecisionTree(results_text_file, output_directory, scorersKey, algoData, classifierSeed, trainingSeed):
    """
    Analyze a DecisionTreeClassifier for classification and report results.

    This function performs analysis on a DecisionTreeClassifier for classification and reports various evaluation metrics
    to the results text file.

    Args:
        results_text_file (file): The file to write the results to.
        output_directory (str): The directory where additional output files will be saved.
        scorersKey (dict): A dictionary of scoring functions.
        algoData (TrainTestvalidationData): An object containing data for analysis.
        classifierSeed (int): The random seed for the classifier.
        trainingSeed (int): The random seed for data splitting during training.

    Returns:
        None
    """
    param_decisiontree = {
        'max_depth': range(1, 30),
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'class_weight':['balanced', 'balanced_subsample']
    }

    results_text_file.write("\n---------------------------DecisionTreeClassifier---------------------------\n")
    print("DecisionTreeClassifier")
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=trainingSeed)
    clf = GridSearchCV(DecisionTreeClassifier(random_state=classifierSeed), param_decisiontree, cv=stratified_kfold,
                       scoring=scorersKey, n_jobs=-1, refit='weighted_f1',
                       error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)
    # print("FEATURES:")
    # for i,v, in enumerate(clf.best_estimator_.feature_importances_):
    #     print('Feature: %0d, Score: %.5f' % (i,v))

    joblib.dump(clf, '%s/model_DecisionTreeClassifier.pkl' % output_directory)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_DecisionTreeClassifier.csv" % output_directory)

    presult_f1 = permutation_importance(clf.best_estimator_, algoData.X_train, algoData.y_train, scoring='f1_weighted')
    results_text_file.write("f1_weighted importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_f1.importances):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    results_text_file.write("mean f1_weighted importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_f1.importances_mean):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=value))
    results_text_file.write("\n")

    presult_balanced = permutation_importance(clf.best_estimator_, algoData.X_train, algoData.y_train, scoring='balanced_accuracy')
    results_text_file.write("balanced_accuracy importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_balanced.importances):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    results_text_file.write("mean balanced_accuracy importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_balanced.importances_mean):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=value))
    results_text_file.write("\n")

    presult_accuracy = permutation_importance(clf.best_estimator_, algoData.X_train, algoData.y_train, scoring='accuracy')
    results_text_file.write("accuracy importances\n")
    for feature, value in zip(algoData.X_train.columns, presult_accuracy.importances):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("\n")
    results_text_file.write("The scores are computed on the full evaluation set.")
    results_text_file.write("\n")

    # Get the best estimator
    best_model = clf.best_estimator_

    y_validation_pred = best_model.predict(algoData.X_validation)
    
    # Calculate accuracy
    accuracy = accuracy_score(algoData.y_validation, y_validation_pred)

    # Create DataFrame with actual and predicted labels
    validation_results = pd.DataFrame({
        'Actual_label': algoData.y_validation,
        'Predicted_Label': y_validation_pred,
        'Accuracy': [accuracy] * len(algoData.y_validation)
    })

    # Save the DataFrame to a CSV file
    validation_results.to_csv('validation_results.csv', index=False)

    # Perform cross-validation using cross_validate()
    cross_val_results = cross_validate(best_model, algoData.X_train, algoData.y_train, cv=stratified_kfold, scoring=scorersKey)

    # Access the cv_results_ specifically for the best estimator
    best_estimator_cv_results = {key: value[clf.best_index_] for key, value in clf.cv_results_.items()}

    # Write the best estimator's cv_results to a text file
    results_text_file.write('--- Best Estimator CV Results ---\n')
    for key, value in best_estimator_cv_results.items():
        results_text_file.write(f'{key}: {value}\n')
    results_text_file.write("\n")

    # Write cross-validation results to the text file
    results_text_file.write('--- Cross-Validation Results ---\n')
    results_text_file.write('Test Scores:\n')
    for key, values in cross_val_results.items():
        results_text_file.write(f'{key}: {values}\n')

    # Calculate and write mean and std for each metric
    for metric in scorersKey:
        mean_score = cross_val_results[f'test_{metric}'].mean()
        std_score = cross_val_results[f'test_{metric}'].std()
        results_text_file.write(f'{metric} (mean): {mean_score}\n')
        results_text_file.write(f'{metric} (std): {std_score}\n')

    results_text_file.write("\n")

    y_true, y_pred = algoData.y_test, best_model.predict(algoData.X_test)
    results_text_file.write('--- Classification Report ---\n')
    results_text_file.write(classification_report(y_true, y_pred))
    results_text_file.write("\n")

    results_text_file.write('balanced_accuracy_score :')
    results_text_file.write(str(balanced_accuracy_score(y_true, y_pred)))
    results_text_file.write("\n")
    results_text_file.write('f1_score (macro) :')
    results_text_file.write(str(f1_score(y_true, y_pred, average='macro')))
    results_text_file.write("\n")
    results_text_file.write('f1_score (micro) :')
    results_text_file.write(str(f1_score(y_true, y_pred, average='micro')))
    results_text_file.write("\n")
    results_text_file.write('f1_score (weighted) :')
    results_text_file.write(str(f1_score(y_true, y_pred, average='weighted')))
    results_text_file.write("\n")
    results_text_file.write('matthews_corrcoef :')
    results_text_file.write(str(matthews_corrcoef(y_true, y_pred)))
    results_text_file.write("\n")
    results_text_file.flush()
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test, 'DecisionTreeClassifier',
                                   output_directory)
    cm = confusion_matrix(y_true, y_pred, labels=algoData.labels)
    utils.print_cm(cm, algoData.labels, classifier='DecisionTreeClassifier', output_directory=output_directory)
    results_text_file.write("\n------------------------------------------------------\n")
