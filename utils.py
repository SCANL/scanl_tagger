import os

import pandas as pd


def print_prediction_results(inputs, predictions, labels, classifier, output_directory):
    dfObj = pd.DataFrame(columns=['Id', 'PredictedLabel', 'ActualLabel'])
    for input, prediction, label in zip(inputs, predictions, labels):
        # if prediction != label:
        dfObj = dfObj.append({'Id': input, 'PredictedLabel': prediction, 'ActualLabel': label}, ignore_index=True)

    dfObj.to_csv("%s/prediction_results_%s.csv" % (output_directory, classifier))


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None, classifier=None,
             output_directory=None):
    """pretty print for confusion matrixes"""

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    filename = '%s/confusion_matrix.txt' % output_directory
    if os.path.exists(filename):
        append_write = 'a'
    else:
        append_write = 'w'
    results_text_file = open(filename, append_write)

    results_text_file.write("\n=======================================================\n")
    results_text_file.write("\n%s\n" % classifier)

    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    results_text_file.write("    " + str(fst_empty_cell) + "\t")
    # End CHANGES

    for label in labels:
        results_text_file.write("%{0}s\t".format(columnwidth) % label)

    results_text_file.write("\n")
    # Print rows
    for i, label1 in enumerate(labels):
        results_text_file.write("    %{0}s\t".format(columnwidth) % label1)
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            results_text_file.write(str(cell) + "\t")
        results_text_file.write("\n")
