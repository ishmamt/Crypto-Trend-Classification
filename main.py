import os
import pandas as pd

from Models.logistic_regression import Logistic_Regression
from Models.random_forest import Random_Forest
from Models.xgboost import XG_Boost
from Models.lightgbm import LightGBM
from Models.decision_tree import Decision_Tree
from Models.svm import SVM
from Models.knn import KNN


data_dir = os.path.join("Data", "Dataset")
datsets = ["data_train.csv", "data_test.csv", "label_train.csv", "label_test.csv"]

X_train = pd.read_csv(os.path.join(data_dir, datsets[0]))
X_test = pd.read_csv(os.path.join(data_dir, datsets[1]))
y_train = pd.read_csv(os.path.join(data_dir, datsets[2]))
y_test = pd.read_csv(os.path.join(data_dir, datsets[3]))

models = [Logistic_Regression(),
          Random_Forest(),
          XG_Boost(),
          LightGBM(),
          Decision_Tree(),
          SVM(),
          KNN()]

scores = list()
for model in models:
    model.train(X_train, y_train)
    scores.append(model.evaluate(X_test, y_test))

# Create a DataFrame from the scores and sort it by the best AUC/ROC score
performance_table = pd.DataFrame(scores, columns=["Classifier", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"])
# performance_table = performance_table.sort_values(by="ROC AUC", ascending=False).reset_index(drop=True)
print(performance_table)

performance_table.to_csv(os.path.join("Results", "result.csv"))