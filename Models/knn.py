from sklearn.neighbors import KNeighborsClassifier

# from model import Model
from evals import calculate_eval_metrics


class KNN():
    """
    Class for KNN model.
    Attributes:
    name (str): Name of the model kept for creating the results table.
    model (RandomForestClassifier): The model object that will be trained and evaluated.
    """

    def __init__(self, name="K-Nearest Neighbors", random_state=42):
        """
        This method gets called for each new instace of the class.

        Parameters:
        name (str): Name of the model kept for creating the results table.
        random_state (int): A random seed for reproducibility (default is 42).
        """

        self.name = name
        self.model = KNeighborsClassifier()

    
    def train(self, X_train, y_train):
        """
        Trains the model on the given training data.
        
        Parameters:
        X_train (pd.DataFrame): Training data.
        y_train (pd.DataFrame): Training labels.
        """

        self.model.fit(X_train, y_train)


    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on the given test data and returns the results.
        
        Parameters:
        X_test (pd.DataFrame): Testing data.
        y_test (pd.DataFrame): Testing labels.

        Returns:
        model_performance (list): A list containing Accuracy, Precision, Recall, F1 Score and ROC AUC scores.
        """

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate the evaluation metrics and return the values.
        model_performance = calculate_eval_metrics(y_pred, y_proba, y_test, self.name)

        return model_performance