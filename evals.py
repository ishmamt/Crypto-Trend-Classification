from tabulate import tabulate
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def calculate_eval_metrics(y_pred, y_proba, y_test, name):
    """
    Returns the evaluationmetrics values after calculations.

    Parameters:
    y_pred (np.Array): The model predictions.
    y_proba (np.Array): The model prediction probabilities.
    y_test (pd.Series): The accurate labels of the test dataset.
    name (str): Name of the model kept for creating the results table.

    Returns:
    model_performance (list): A list containing Accuracy, Precision, Recall, F1 Score and ROC AUC scores.
    """

    model_performance = list()
    model_performance.append(name)

    # Calculate the Accuracy, Precision, Recall, F1 Score and ROC AUC scores and append to the list
    model_performance.append(accuracy_score(y_test, y_pred))
    model_performance.append(precision_score(y_test, y_pred))
    model_performance.append(recall_score(y_test, y_pred))
    model_performance.append(f1_score(y_test, y_pred))
    model_performance.append(roc_auc_score(y_test, y_proba))

    return model_performance


def tabulate_results(model_performance):
    """
    Creates a clean tabular report for printing.

    Parameters:
    model_performance (list): A list containing Accuracy, Precision, Recall, F1 Score and ROC AUC scores.

    Returns:
    tabular_report_data (str): Model performance metrics in a clean tabular report format for printing.
    """

    # Create a tabular report
    report_data = [["Metric", "Value"],
                ["Name", model_performance[0]],
                ["Accuracy", model_performance[1]],
                ["Precision", model_performance[2]],
                ["Recall", model_performance[3]],
                ["F1 Score", model_performance[4]],
                ["ROC AUC", model_performance[5]]]

    return tabulate(report_data, tablefmt="fancy_outline")
