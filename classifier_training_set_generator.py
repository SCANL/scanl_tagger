import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier
import numpy as np

#RANDOM_CLASSIFIER_SEED = 5545
RANDOM_CLASSIFIER_SEED = 1269

def build_datasets(X, y, text_column, output_directory):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1.0, test_size=0.25, random_state=42)
    # X_train_original = X_train.copy(deep=True)
    # X_test_original = X_test.copy(deep=True)

    #unique, counts = np.unique(y_train, return_counts=True)
    # print(" --  -- Distribution of labels in training --  -- ")
    # print(dict(zip(unique, counts)))

    return X, y.values.ravel()


def perform_classification(X, y, text_column, results_text_file, output_directory):
    X_train, y_train = build_datasets(X, y, "", output_directory)

    param_randomforest = {
        'n_estimators': [350],
        'max_depth': range(30, 100),
        'criterion': ['gini', 'entropy'],
        'bootstrap': [True]
    }

    param_decisiontree = {
        'max_depth': range(1, 30),
        'criterion': ['gini', 'entropy']
    }

    scorers = {
        'f1_score_micro': make_scorer(f1_score, average='micro'),
    }

    labels = np.unique(y_train, return_counts=False)

    # https://scikit-learn.org/stable/modules/model_evaluation.html
    scoring = ('accuracy', 'balanced_accuracy', 'f1_micro', 'precision_micro', 'recall_micro')

    for key in scorers:
        results_text_file.write("\n%s" % key)
        results_text_file.write("\n------------------------------")
        results_text_file.write("\n------------------------------")

        results_text_file.write("\nDecisionTreeClassifier\n")
        print("DecisionTreeClassifier")
        clf = DecisionTreeClassifier(random_state=RANDOM_CLASSIFIER_SEED, max_depth=9, criterion='entropy')
        clf.fit(X_train, y_train)
        joblib.dump(clf, '%s/model_DecisionTreeClassifier.pkl' % output_directory)


        results_text_file.write("\nRandomForestClassifier\n")
        print("RandomForestClassifier")
        clf = RandomForestClassifier(random_state=RANDOM_CLASSIFIER_SEED, n_estimators=250, criterion="gini", max_depth=83, bootstrap=True)
        clf.fit(X_train, y_train)
        joblib.dump(clf, '%s/model_RandomForestClassifier.pkl' % output_directory)