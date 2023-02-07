import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier, \
    GradientBoostingClassifier
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import numpy as np
import pandas as pd
import scipy as sp
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.metrics import classification_report_imbalanced
import random
import utils
from enum import Enum


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


# 10, 25, 500 (t), 300(t), 2704875 484565 500 501
# 1269 5545 1269
# RANDOM_TRAIN_SEED = 484565
# RANDOM_CLASSIFIER_SEED = 5545
# RANDOM_TRAIN_SEED = 236373
# RANDOM_TRAIN_SEED = round(random.random() * 1000000)
# print("Random Training Seed: " + str(RANDOM_TRAIN_SEED))
RANDOM_CLASSIFIER_SEED = 1269


def build_datasets(X, y, text_column, output_directory, trainingSeed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=trainingSeed)
    X_train_original = X_train.copy(deep=True)
    X_test_original = X_test.copy(deep=True)
    # if text_column:
    #     vec = TfidfVectorizer()
    #     #vec = CountVectorizer(ngram_range=(1, 3))
    #     vec = TfidfVectorizer(max_features=2000, min_df=3, sublinear_tf=True, ngram_range=(1, 1))

    #     X_train = sp.sparse.hstack((vec.fit_transform(X_train[text_column]),
    #                                 X_train[X_train.columns.difference([text_column])].values),
    #                             format='csr')
    #     X_test = sp.sparse.hstack((vec.transform(X_test[text_column]),
    #                             X_test[X_test.columns.difference([text_column])].values),
    #                             format='csr')
    #     joblib.dump(vec, '%s/model_TfidfVectorizer.pkl' % output_directory)

    # # uncomment for rebalancing
    # #sm = SMOTE('minority',random_state=RANDOM_CLASSIFIER_SEED)
    # # sm = SMOTE('minority', kind='regular', random_state=RANDOM_CLASSIFIER_SEED)
    # # X_train, y_train = sm.fit_sample(X_train, y_train.values.ravel())

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

    # estimators_VotingClassifier = VotingClassifier(estimators=[
    #     ('svc', SVC(C=1.989999999999999, gamma="scale", kernel="linear")),
    #     ('rf', RandomForestClassifier(bootstrap=False, criterion="gini", max_depth=78, n_estimators=500)),
    #     ('lr', LogisticRegression(multi_class='auto', penalty="l1", solver="liblinear")),
    #     ('lsvc', LinearSVC(C=0.7499999999999997, dual=False, loss="squared_hinge", multi_class="ovr", penalty="l1")),
    #     ('dt', DecisionTreeClassifier(criterion="gini", max_depth=75)),
    #     ('bnb', BernoulliNB(alpha=0.16999999999999998)),
    #     ('mnb', MultinomialNB(alpha=2.629999999999999)),
    #     ('knn', KNeighborsClassifier(n_neighbors=69)),
    #     ('bag', BaggingClassifier(bootstrap=True, n_estimators=700))
    # ])

    scorers = {
        # 'precision_score_micro': make_scorer(precision_score,  average='micro'),
        # 'precision_score_macro': make_scorer(precision_score, average='macro'),
        # 'recall_score_micro': make_scorer(recall_score, average='micro'),
        # 'recall_score_macro': make_scorer(recall_score, average='macro'),
        # 'accuracy_score': make_scorer(accuracy_score),
        # 'f1_score_weighted': make_scorer(f1_score, average='weighted'),
        # 'f1_score_macro': make_scorer(f1_score,  average='macro'),
        # 'hamming_loss': make_scorer(hamming_loss)
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
            if algorithm == Algorithm.XGBOOST:
                analyzeXGBoost(results_text_file, output_directory, scorers[key], scoring, algoData)
            if algorithm == Algorithm.LOGISTIC:
                analyzeLogistic(results_text_file, output_directory, scorers[key], scoring, algoData)
            if algorithm == Algorithm.SVC:
                analyzeSVC(results_text_file, output_directory, scorers[key], scoring, algoData)
            if algorithm == Algorithm.LINEAR_SVC:
                analyzeLinearSVC(results_text_file, output_directory, scorers[key], scoring, algoData)
            if algorithm == Algorithm.ADA_BOOST:
                analyzeAdaBoost(results_text_file, output_directory, scorers[key], algoData)
            if algorithm == Algorithm.ADA_BOOST_FOREST:
                analyzeAdaBoostForest(results_text_file, output_directory, scorers[key], algoData)
            if algorithm == Algorithm.BAGGING:
                analyzeBagging(results_text_file, output_directory, scorers[key], scoring, algoData)
            if algorithm == Algorithm.GRADIENT_BOOSTING:
                analyzeGradientBoosting(results_text_file, output_directory, scorers[key], algoData)
            if algorithm == Algorithm.MULTI_NB:
                analyzeMultiNB(results_text_file, output_directory, scorers[key], scoring, algoData)
            if algorithm == Algorithm.BERNOULLI:
                analyzeBernoulli(results_text_file, output_directory, scorers[key], scoring, algoData)
            if algorithm == Algorithm.K_NEIGHBORS:
                analyzeKN(results_text_file, output_directory, scorers[key], scoring, algoData)

        # results_text_file.write("\n%s" % key)
        # results_text_file.write("\n------------------------------")
        # results_text_file.write("\n------------------------------")

        # results_text_file.write("\nVotingClassifier\n")
        # print("VotingClassifier")
        # clf = GridSearchCV(estimator=estimators_VotingClassifier, param_grid={}, cv=5, scoring=scorers[key], n_jobs=-1,
        #                    error_score=0.0)
        # clf.fit(X_train, y_train)
        # joblib.dump(clf, '%s/model_VotingClassifier.pkl' % output_directory)
        # results_text_file.write("Best parameters set found on development set:")
        # results_text_file.write("\n")
        # results_text_file.write(json.dumps(clf.best_params_))
        # results_text_file.write("\n")
        # pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_VotingClassifier.csv" % output_directory)
        # results_text_file.write("Detailed classification report:")
        # results_text_file.write("\n")
        # results_text_file.write("The model is trained on the full development set.")
        # results_text_file.write("The scores are computed on the full evaluation set.")
        # results_text_file.write("\n")
        # y_true, y_pred = y_test, clf.predict(X_test)
        # results_text_file.write(classification_report(y_true, y_pred))
        # results_text_file.write("\n")
        # results_text_file.write('balanced_accuracy_score :')
        # results_text_file.write(str(balanced_accuracy_score(y_true, y_pred)))
        # results_text_file.write("\n")
        # results_text_file.write('f1_score (macro) :')
        # results_text_file.write(str(f1_score(y_true, y_pred, average='macro')))
        # results_text_file.write("\n")
        # results_text_file.write('f1_score (micro) :')
        # results_text_file.write(str(f1_score(y_true, y_pred, average='micro')))
        # results_text_file.write("\n")
        # results_text_file.write('f1_score (weighted) :')
        # results_text_file.write(str(f1_score(y_true, y_pred, average='weighted')))
        # results_text_file.write("\n")
        # results_text_file.write('matthews_corrcoef :')
        # results_text_file.write(str(matthews_corrcoef(y_true, y_pred)))
        # results_text_file.write("\n")
        # results_text_file.flush()
        # utils.print_prediction_results(X_test_original.index, y_pred, y_test, 'VotingClassifier', output_directory)
        # cm = confusion_matrix(y_true, y_pred, labels=labels)
        # utils.print_cm(cm, labels, classifier='VotingClassifier', output_directory=output_directory)

        # results_text_file.write("\nGaussianNB\n")
        # print("GaussianNB")
        # clf = GridSearchCV(GaussianNB(), param_grid={}, cv=10, scoring=scorers[key] ,n_jobs=-1, error_score=0.0)
        # clf.fit(X_train.todense(), y_train)
        # joblib.dump(clf, '%s/model_GaussianNB.pkl' % output_directory)
        # results_text_file.write("Best parameters set found on development set:")
        # results_text_file.write("\n")
        # results_text_file.write(json.dumps(clf.best_params_))
        # results_text_file.write("\n")
        # pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_GaussianNB.csv" % output_directory)
        # results_text_file.write("Detailed classification report:")
        # results_text_file.write("\n")
        # results_text_file.write("The model is trained on the full development set.")
        # results_text_file.write("The scores are computed on the full evaluation set.")
        # results_text_file.write("\n")
        # cv_results = cross_validate(clf, X_test.todense(), y_test, cv=10, scoring=scoring)
        # results_text_file.write('cv_results :\n')
        # results_text_file.write(str(cv_results))
        # results_text_file.write("\n")
        # y_true, y_pred = y_test, clf.predict(X_test.todense())
        # results_text_file.write(classification_report(y_true, y_pred))
        # results_text_file.write("\n")
        # results_text_file.write('balanced_accuracy_score :')
        # results_text_file.write(str(balanced_accuracy_score(y_true, y_pred)))
        # results_text_file.write("\n")
        # results_text_file.write('f1_score (macro) :')
        # results_text_file.write(str(f1_score(y_true, y_pred, average='macro')))
        # results_text_file.write("\n")
        # results_text_file.write('f1_score (micro) :')
        # results_text_file.write(str(f1_score(y_true, y_pred, average='micro')))
        # results_text_file.write("\n")
        # results_text_file.write('f1_score (weighted) :')
        # results_text_file.write(str(f1_score(y_true, y_pred, average='weighted')))
        # results_text_file.write("\n")
        # results_text_file.write('matthews_corrcoef :')
        # results_text_file.write(str(matthews_corrcoef(y_true, y_pred)))
        # results_text_file.write("\n")
        # results_text_file.flush()
        # utils.print_prediction_results(X_test_original.index, y_pred, y_test, 'GaussianNB', output_directory)
        # cm = confusion_matrix(y_true, y_pred, labels=labels)
        # utils.print_cm(cm, labels, classifier='GaussianNB', output_directory=output_directory)


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
        'max_depth': range(1, 60),
        'criterion': ['gini', 'entropy']
    }

    results_text_file.write("\n---------------------------DecisionTreeClassifier---------------------------\n")
    print("DecisionTreeClassifier")
    clf = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_CLASSIFIER_SEED), param_decisiontree, cv=5,
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
    # for feature, value in zip(algoData.X.columns, presult_f1.importances):
    #     results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
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
    # for feature, value in zip(algoData.X.columns, presult_accuracy.importances):
    #     results_text_file.write("{feature},{value}\n".format(feature=feature, value=','.join(str(v) for v in value)))
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

    # results_text_file.write("Detailed classification report:")
    # results_text_file.write("\n")
    # results_text_file.write("The model is trained on the full development set.")
    # results_text_file.write("The scores are computed on the full evaluation set.")
    # results_text_file.write("\n")
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
    # results_text_file.write("\n")
    # results_text_file.write('balanced_accuracy_score :')
    # results_text_file.write(str(balanced_accuracy_score(y_true, y_pred)))
    # results_text_file.write("\n")
    # results_text_file.write('f1_score (macro) :')
    # results_text_file.write(str(f1_score(y_true, y_pred, average='macro')))
    # results_text_file.write("\n")
    # results_text_file.write('f1_score (micro) :')
    # results_text_file.write(str(f1_score(y_true, y_pred, average='micro')))
    # results_text_file.write("\n")
    # results_text_file.write('f1_score (weighted) :')
    # results_text_file.write(str(f1_score(y_true, y_pred, average='weighted')))
    # results_text_file.write("\n")
    # results_text_file.write('matthews_corrcoef :')
    # results_text_file.write(str(matthews_corrcoef(y_true, y_pred)))
    # results_text_file.write("\n")
    results_text_file.flush()
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test, 'DecisionTreeClassifier',
                                   output_directory)
    cm = confusion_matrix(y_true, y_pred, labels=algoData.labels)
    utils.print_cm(cm, algoData.labels, classifier='DecisionTreeClassifier', output_directory=output_directory)
    results_text_file.write("\n------------------------------------------------------\n")


def analyzeXGBoost(results_text_file, output_directory, scorersKey, scoring, algoData):
    param_xgboost = {
        "learning_rate": np.arange(0.1, 1.0, 0.01),
        "min_child_weight": range(1, 31),
        "n_estimators": [5, 10],
        "max_depth": range(1, 10),
        'n_neighbors': range(1, 10)
    }

    results_text_file.write("\nXGBoost\n")
    print("XGBoost")
    clf = GridSearchCV(XGBClassifier(), param_xgboost, cv=5, scoring=scorersKey, n_jobs=-1, error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)
    joblib.dump(clf, '%s/model_XGBoost.pkl' % output_directory)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_XGBoost.csv" % output_directory)
    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("The scores are computed on the full evaluation set.")
    results_text_file.write("\n")
    cv_results = cross_validate(clf, algoData.X_test, algoData.y_test, cv=5, scoring=scoring)
    results_text_file.write('cv_results :\n')
    results_text_file.write(str(cv_results))
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
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test, 'XGBoost', output_directory)
    cm = confusion_matrix(y_true, y_pred, labels=algoData.labels)
    utils.print_cm(cm, algoData.labels, classifier='XGBoost', output_directory=output_directory)


def analyzeLogistic(results_text_file, output_directory, scorersKey, scoring, algoData):
    param_logisticregression = {
        'solver': ['newton-cg', 'sag', 'saga', 'lbfgs', 'liblinear'],
        'penalty': ['l1', 'l2']
    }

    results_text_file.write("\nLogisticRegression\n")
    print("LogisticRegression")
    clf = GridSearchCV(LogisticRegression(multi_class='auto'), param_logisticregression, cv=10, scoring=scorersKey,
                       n_jobs=-1,
                       error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)
    joblib.dump(clf, '%s/model_LogisticRegression.pkl' % output_directory)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_LogisticRegression.csv" % output_directory)
    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("The scores are computed on the full evaluation set.")
    results_text_file.write("\n")
    cv_results = cross_validate(clf, algoData.X_test, algoData.y_test, cv=10, scoring=scoring)
    results_text_file.write('cv_results :\n')
    results_text_file.write(str(cv_results))
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
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test, 'LogisticRegression',
                                   output_directory)
    cm = confusion_matrix(y_true, y_pred, labels=algoData.labels)
    utils.print_cm(cm, algoData.labels, classifier='LogisticRegression', output_directory=output_directory)


def analyzeSVC(results_text_file, output_directory, scorersKey, scoring, algoData):
    param_svc = {
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': np.arange(0.1, 5.0, 0.01)
    }

    results_text_file.write("\nSVC\n")
    print("SVC")
    clf = GridSearchCV(SVC(), param_svc, cv=10, scoring=scorersKey, n_jobs=-1, error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)
    joblib.dump(clf, '%s/model_SVC.pkl' % output_directory)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_SVC.csv" % output_directory)
    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("The scores are computed on the full evaluation set.")
    results_text_file.write("\n")
    cv_results = cross_validate(clf, algoData.X_test, algoData.y_test, cv=10, scoring=scoring)
    results_text_file.write('cv_results :\n')
    results_text_file.write(str(cv_results))
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
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test, 'SVC', output_directory)
    cm = confusion_matrix(y_true, y_pred, labels=algoData.labels)
    utils.print_cm(cm, algoData.labels, classifier='SVC', output_directory=output_directory)


def analyzeLinearSVC(results_text_file, output_directory, scorersKey, scoring, algoData):
    param_linearsvc = {
        'penalty': ['l1', 'l2'],
        'loss': ['hinge', 'squared_hinge'],
        'dual': [True, False],
        'multi_class': ['ovr', 'crammer_singer'],
        'C': np.arange(0.1, 5.0, 0.01)
    }

    results_text_file.write("\nLinearSVC\n")
    print("LinearSVC")
    clf = GridSearchCV(LinearSVC(), param_linearsvc, cv=10, scoring=scorersKey, n_jobs=-1, error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)
    joblib.dump(clf, '%s/model_LinearSVC.pkl' % output_directory)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_LinearSVC.csv" % output_directory)
    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("The scores are computed on the full evaluation set.")
    results_text_file.write("\n")
    cv_results = cross_validate(clf, algoData.X_test, algoData.y_test, cv=10, scoring=scoring)
    results_text_file.write('cv_results :\n')
    results_text_file.write(str(cv_results))
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
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test, 'LinearSVC',
                                   output_directory)
    cm = confusion_matrix(y_true, y_pred, labels=algoData.labels)
    utils.print_cm(cm, algoData.labels, classifier='LinearSVC', output_directory=output_directory)


def analyzeAdaBoost(results_text_file, output_directory, scorersKey, algoData):
    param_adaboost = {
        'n_estimators': [5, 10, 50, 100, 300, 500, 700, 725, 740, 750, 760, 780, 790, 800, 900, 1000],
        'algorithm': ['SAMME', 'SAMME.R'],
        'base_estimator__max_depth': range(1, 101),
        'base_estimator__criterion': ['gini', 'entropy']
    }

    results_text_file.write("\nAdaBoostClassifier\n")
    print("AdaBoostClassifier")
    clf = GridSearchCV(AdaBoostClassifier(base_estimator=DecisionTreeClassifier()), param_adaboost, cv=5,
                       scoring=scorersKey, n_jobs=-1, error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)
    joblib.dump(clf, '%s/model_AdaBoostClassifier.pkl' % output_directory)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_AdaBoostClassifier.csv" % output_directory)
    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("The scores are computed on the full evaluation set.")
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
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test, 'AdaBoostClassifier',
                                   output_directory)


def analyzeAdaBoostForest(results_text_file, output_directory, scorersKey, algoData):
    param_adaboost_randomforest = {
        'n_estimators': [740, 750, 760, 780, 1000],
        'algorithm': ['SAMME', 'SAMME.R'],
        'base_estimator__n_estimators': [500, 700, 900, 1000, 1100],
        'base_estimator__max_depth': range(40, 71),
        'base_estimator__criterion': ['gini'],
        'base_estimator__bootstrap': [False]
    }

    results_text_file.write("\nAdaBoostClassifier_RandomForest\n")
    print("AdaBoostClassifier_RandomForest")
    clf = GridSearchCV(AdaBoostClassifier(base_estimator=RandomForestClassifier()), param_adaboost_randomforest, cv=5,
                       scoring=scorersKey, n_jobs=-1, error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)
    joblib.dump(clf, '%s/model_AdaBoostClassifier_RandomForest.pkl' % output_directory)
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_AdaBoostClassifier_RandomForest.csv" % output_directory)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("The scores are computed on the full evaluation set.")
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
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test,
                                   'AdaBoostClassifier_RandomForest', output_directory)


def analyzeBagging(results_text_file, output_directory, scorersKey, scoring, algoData):
    param_bagging = {
        'n_estimators': [5, 10, 50, 75, 100, 150, 300, 500],
        'bootstrap': [True, False]
    }

    results_text_file.write("\nBaggingClassifier\n")
    print("BaggingClassifier")
    clf = GridSearchCV(BaggingClassifier(), param_bagging, cv=10, scoring=scorersKey, n_jobs=-1, error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)
    joblib.dump(clf, '%s/model_BaggingClassifier.pkl' % output_directory)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_BaggingClassifier.csv" % output_directory)
    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("The scores are computed on the full evaluation set.")
    results_text_file.write("\n")
    cv_results = cross_validate(clf, algoData.X_test, algoData.y_test, cv=10, scoring=scoring)
    results_text_file.write('cv_results :\n')
    results_text_file.write(str(cv_results))
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
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test, 'BaggingClassifier',
                                   output_directory)
    cm = confusion_matrix(y_true, y_pred, labels=algoData.labels)
    utils.print_cm(cm, algoData.labels, classifier='BaggingClassifier', output_directory=output_directory)


def analyzeGradientBoosting(results_text_file, output_directory, scorersKey, algoData):
    param_gradientboost = {
        'loss': ['deviance', 'exponential'],
        'learning_rate': [0.5, 0.25, 0.1, 0.05],
        'n_estimators': [5, 10, 50, 100, 300, 500, 700, 725, 740, 750, 760, 780, 790, 800, 900, 1000],
        'max_depth': range(1, 51)
    }

    results_text_file.write("\nGradientBoosting\n")
    print("GradientBoosting")
    clf = GridSearchCV(GradientBoostingClassifier(), param_gradientboost, cv=5, scoring=scorersKey, n_jobs=-1,
                       error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_GradientBoosting.csv" % output_directory)
    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("The scores are computed on the full evaluation set.")
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
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test, 'GradientBoosting',
                                   output_directory)


def analyzeMultiNB(results_text_file, output_directory, scorersKey, scoring, algoData):
    param_multinomialnb = {
        'alpha': np.arange(0.1, 5.0, 0.01)
    }

    results_text_file.write("\nMultinomialNB\n")
    print("MultinomialNB")
    clf = GridSearchCV(MultinomialNB(), param_multinomialnb, cv=10, scoring=scorersKey, n_jobs=-1, error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)
    joblib.dump(clf, '%s/model_MultinomialNB.pkl' % output_directory)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_MultinomialNB.csv" % output_directory)
    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("The scores are computed on the full evaluation set.")
    results_text_file.write("\n")
    cv_results = cross_validate(clf, algoData.X_test, algoData.y_test, cv=10, scoring=scoring)
    results_text_file.write('cv_results :\n')
    results_text_file.write(str(cv_results))
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
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test, 'MultinomialNB',
                                   output_directory)
    cm = confusion_matrix(y_true, y_pred, labels=algoData.labels)
    utils.print_cm(cm, algoData.labels, classifier='MultinomialNB', output_directory=output_directory)


def analyzeBernoulli(results_text_file, output_directory, scorersKey, scoring, algoData):
    param_bernoullinb = {
        'alpha': np.arange(0.1, 5.0, 0.01)
    }

    results_text_file.write("\nBernoulliNB\n")
    print("BernoulliNB")
    clf = GridSearchCV(BernoulliNB(), param_bernoullinb, cv=10, scoring=scorersKey, n_jobs=-1, error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)
    joblib.dump(clf, '%s/model_BernoulliNB.pkl' % output_directory)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_BernoulliNB.csv" % output_directory)
    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("The scores are computed on the full evaluation set.")
    results_text_file.write("\n")
    cv_results = cross_validate(clf, algoData.X_test, algoData.y_test, cv=10, scoring=scoring)
    results_text_file.write('cv_results :\n')
    results_text_file.write(str(cv_results))
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
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test, 'BernoulliNB',
                                   output_directory)
    cm = confusion_matrix(y_true, y_pred, labels=algoData.labels)
    utils.print_cm(cm, algoData.labels, classifier='BernoulliNB', output_directory=output_directory)


def analyzeKN(results_text_file, output_directory, scorersKey, scoring, algoData):
    param_knn = {
        'n_neighbors': range(1, 101)
    }

    results_text_file.write("\nKNeighborsClassifier\n")
    print("KNeighborsClassifier")
    clf = GridSearchCV(KNeighborsClassifier(), param_knn, cv=10, scoring=scorersKey, n_jobs=-1, error_score=0.0)
    clf.fit(algoData.X_train, algoData.y_train)
    joblib.dump(clf, '%s/model_KNN.pkl' % output_directory)
    results_text_file.write("Best parameters set found on development set:")
    results_text_file.write("\n")
    results_text_file.write(json.dumps(clf.best_params_))
    results_text_file.write("\n")
    pd.DataFrame(clf.cv_results_).to_csv("%s/cv_results_KNeighborsClassifier.csv" % output_directory)
    results_text_file.write("Detailed classification report:")
    results_text_file.write("\n")
    results_text_file.write("The model is trained on the full development set.")
    results_text_file.write("The scores are computed on the full evaluation set.")
    results_text_file.write("\n")
    cv_results = cross_validate(clf, algoData.X_test, algoData.y_test, cv=10, scoring=scoring)
    results_text_file.write('cv_results :\n')
    results_text_file.write(str(cv_results))
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
    utils.print_prediction_results(algoData.X_test_original.index, y_pred, algoData.y_test, 'KNeighborsClassifier',
                                   output_directory)
    cm = confusion_matrix(y_true, y_pred, labels=algoData.labels)
    utils.print_cm(cm, algoData.labels, classifier='KNeighborsClassifier', output_directory=output_directory)


if __name__ == '__main__':
    perform_classification(1, 1, 1, 1, 1)
