import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evals import calculate_eval_metrics, tabulate_results


class Logistic_Regression():
    """
    Class for logistic regression model.
    Attributes:
    name (str): Name of the model kept for creating the results table.
    model (LogisticRegression): The model object that will be trained and evaluated.
    """

    def __init__(self, data_dir, datasets, name="Logistic Regression", random_state=42):
        """
        This method gets called for each new instace of the class.

        Parameters:
        data_dir (str): Path to the dataset.
        datasets (list): Filenames of the dataset splits.
        name (str): Name of the model kept for creating the results table.
        random_state (int): A random seed for reproducibility (default is 42).
        """

        self.name = name
        self.model = LogisticRegression(random_state=random_state)

        #For the dataloader
        self.X_train = pd.read_csv(os.path.join(data_dir, datasets[0]))
        self.X_val = pd.read_csv(os.path.join(data_dir, datasets[1]))
        self.X_test = pd.read_csv(os.path.join(data_dir, datasets[2]))
        self.y_train = pd.read_csv(os.path.join(data_dir, datasets[3]))
        self.y_val = pd.read_csv(os.path.join(data_dir, datasets[4]))
        self.y_test = pd.read_csv(os.path.join(data_dir, datasets[5]))

    
    def train(self):
        """
        Trains the model on the given training data.
        """

        self.model.fit(self.X_train, self.y_train)


    def validate(self):
        """
        Validates the model on the given val data and shows the results.
        """

        y_pred = self.model.predict(self.X_val)
        y_proba = self.model.predict_proba(self.X_val)[:, 1]

        # Calculate the evaluation metrics and return the values.
        model_performance = calculate_eval_metrics(y_pred, y_proba, self.y_val, self.name)
        print(tabulate_results(model_performance))


    def test(self):
        """
        Evaluates the model on the given test data and returns the results.

        Returns:
        model_performance (list): A list containing Accuracy, Precision, Recall, F1 Score and ROC AUC scores.
        """

        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Calculate the evaluation metrics and return the values.
        model_performance = calculate_eval_metrics(y_pred, y_proba, self.y_test, self.name)

        return model_performance