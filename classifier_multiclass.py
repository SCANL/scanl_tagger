import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from imblearn.metrics import classification_report_imbalanced
import print_utility_functions as utils
from enum import Enum
import random

class Algorithm(Enum):
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


class AlgoData:

    def __init__(self, X, y, X_train, X_test, y_train, y_test, X_train_original, X_test_original, labels):
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_original = X_train_original
        self.X_test_original = X_test_original
        self.labels = labels

# RANDOM_TRAIN_SEED = 484565
RANDOM_CLASSIFIER_SEED = 1269
print("Classifier seed: " + str(RANDOM_CLASSIFIER_SEED))

def build_datasets(X, y, text_column, output_directory, trainingSeed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=trainingSeed)
    X_train_original = X_train.copy(deep=True)
    X_test_original = X_test.copy(deep=True)

    unique, counts = np.unique(y_train, return_counts=True)
    print(" --  -- Distribution of labels in training --  -- ")
    print(dict(zip(unique, counts)))

    # return X_train, X_test, y_train, y_test.values.ravel(), X_train_original, X_test_original
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel(), X_train_original, X_test_original


def perform_classification(X, y, text_column, results_text_file, output_directory, algorithms_to_use, trainingSeed):
    X_train, X_test, y_train, y_test, X_train_original, X_test_original = build_datasets(X, y, "", output_directory, trainingSeed)
    labels = np.unique(y_train, return_counts=False)

    algoData = AlgoData(X, y, X_train, X_test, y_train, y_test, X_train_original, X_test_original, labels)
    results_text_file.write("Training Seed: %s\n" % trainingSeed)

    scorers = {
        'f1_score_micro': make_scorer(f1_score, average='micro')
    }

    # https://scikit-learn.org/stable/modules/model_evaluation.html
    scoring = ('accuracy', 'balanced_accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted')

    for key in scorers:
        for algorithm in algorithms_to_use:
            if algorithm == Algorithm.RANDOM_FOREST:
                analyzeRandomForest(results_text_file, output_directory, scorers[key], scoring, algoData)
            if algorithm == Algorithm.DECISION_TREE:
                analyzeDecisionTree(results_text_file, output_directory, scorers[key], scoring, algoData)

def analyzeRandomForest(results_text_file, output_directory, scorersKey, scoring, algoData):
    param_randomforest = {
        'n_estimators': [350, 400, 450, 500],
        'max_depth': range(30, 100),
        'criterion': ['gini', 'entropy'],
        'bootstrap': [True]
    }

    results_text_file.write("\n---------------------------RandomForestClassifier---------------------------\n")
    print("RandomForestClassifier")
    clf = GridSearchCV(RandomForestClassifier(random_state=RANDOM_CLASSIFIER_SEED), param_randomforest, cv=5,
                       scoring=scorersKey, n_jobs=-1,
                       error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)

    joblib.dump(clf, '%s/model_RandomForestClassifier.pkl' % output_directory)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_RandomForestClassifier.csv" % output_directory)

    presult_f1 = permutation_importance(clf.best_estimator_, algoData.X, algoData.y, scoring='f1_weighted')
    results_text_file.write("f1_weighted importances\n")
    for feature, value in zip(algoData.X.columns, presult_f1.importances):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    results_text_file.write("mean f1_weighted importances\n")
    for feature, value in zip(algoData.X.columns, presult_f1.importances_mean):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=value))
    results_text_file.write("\n")

    presult_balanced = permutation_importance(clf.best_estimator_, algoData.X, algoData.y, scoring='balanced_accuracy')
    results_text_file.write("balanced_accuracy importances\n")
    for feature, value in zip(algoData.X.columns, presult_balanced.importances):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    results_text_file.write("mean balanced_accuracy importances\n")
    for feature, value in zip(algoData.X.columns, presult_balanced.importances_mean):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=value))
    results_text_file.write("\n")

    presult_accuracy = permutation_importance(clf.best_estimator_, algoData.X, algoData.y, scoring='accuracy')
    results_text_file.write("accuracy importances\n")
    for feature, value in zip(algoData.X.columns, presult_accuracy.importances):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    vec_feature_sum = 0
    results_text_file.write("mean accuracy importances\n")
    for feature, value in zip(algoData.X.columns, presult_accuracy.importances_mean):
        if feature.startswith("VEC"):
            vec_feature_sum += value
        else:
            results_text_file.write("{feature},{value}\n".format(feature=feature, value=value))
    results_text_file.write("{feature},{value}\n".format(feature="VEC", value=vec_feature_sum))
    results_text_file.write("\n")

    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("The scores are computed on the full evaluation set.")
    results_text_file.write("\n")

    cv_results = cross_validate(clf, algoData.X_test, algoData.y_test, cv=5, scoring=scoring)
    results_text_file.write('cv_results :\n')
    for metric, value in cv_results.items():
        results_text_file.write("{metric},{value}\n".format(metric=metric, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")
    y_true, y_pred = algoData.y_test, clf.predict(algoData.X_test)
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
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test, 'RandomForestClassifier',
                                   output_directory)
    cm = confusion_matrix(y_true, y_pred, labels=algoData.labels)
    utils.print_cm(cm, algoData.labels, classifier='RandomForestClassifier', output_directory=output_directory)
    results_text_file.write("\n------------------------------------------------------\n")


def analyzeDecisionTree(results_text_file, output_directory, scorersKey, scoring, algoData):
    param_decisiontree = {
        'max_depth': range(1, 200),
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random']
    }

    results_text_file.write("\n---------------------------DecisionTreeClassifier---------------------------\n")
    print("DecisionTreeClassifier")
    clf = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_CLASSIFIER_SEED), param_decisiontree, cv=6,
                       scoring=scorersKey, n_jobs=-1,
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

    presult_f1 = permutation_importance(clf.best_estimator_, algoData.X, algoData.y, scoring='f1_weighted')
    results_text_file.write("f1_weighted importances\n")
    for feature, value in zip(algoData.X.columns, presult_f1.importances):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    results_text_file.write("mean f1_weighted importances\n")
    # for feature, value in zip(algoData.X.columns, presult_f1.importances_mean):
    #     results_text_file.write("{feature},{value}\n".format(feature=feature, value=value))
    results_text_file.write("\n")

    presult_balanced = permutation_importance(clf.best_estimator_, algoData.X, algoData.y, scoring='balanced_accuracy')
    results_text_file.write("balanced_accuracy importances\n")
    # for feature, value in zip(algoData.X.columns, presult_balanced.importances):
    #     results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    results_text_file.write("mean balanced_accuracy importances\n")
    # for feature, value in zip(algoData.X.columns, presult_balanced.importances_mean):
    #     results_text_file.write("{feature},{value}\n".format(feature=feature, value=value))
    results_text_file.write("\n")

    presult_accuracy = permutation_importance(clf.best_estimator_, algoData.X, algoData.y, scoring='accuracy')
    results_text_file.write("accuracy importances\n")
    for feature, value in zip(algoData.X.columns, presult_accuracy.importances):
        results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")

    vec_feature_sum = 0
    results_text_file.write("mean accuracy importances\n")
    for feature, value in zip(algoData.X.columns, presult_accuracy.importances_mean):
        if feature.startswith("VEC"):
            vec_feature_sum += value
        else:
            results_text_file.write("{feature},{value}\n".format(feature=feature, value=value))
    results_text_file.write("{feature},{value}\n".format(feature="VEC", value=vec_feature_sum))
    results_text_file.write("\n")

    cv_results = cross_validate(clf, algoData.X_test, algoData.y_test, cv=5, scoring=scoring)
    results_text_file.write('cv_results:\n')
    results_text_file.write(str(cv_results))
    for metric, value in cv_results.items():
        results_text_file.write("{metric},{value}\n".format(metric=metric, value=','.join(str(v) for v in value)))
    results_text_file.write("\n")
    y_true, y_pred = algoData.y_test, clf.predict(algoData.X_test)
    results_text_file.write(classification_report(y_true, y_pred))
    results_text_file.write('balanced_accuracy_score :')
    results_text_file.write(str(balanced_accuracy_score(y_true, y_pred)))

    results_text_file.flush()
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test, 'DecisionTreeClassifier',
                                   output_directory)
    cm = confusion_matrix(y_true, y_pred, labels=algoData.labels)
    utils.print_cm(cm, algoData.labels, classifier='DecisionTreeClassifier', output_directory=output_directory)
    results_text_file.write("\n------------------------------------------------------\n")
