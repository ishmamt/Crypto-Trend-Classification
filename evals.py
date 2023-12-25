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