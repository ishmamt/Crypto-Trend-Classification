import os
import pandas as pd

from Models.logistic_regression import Logistic_Regression
from Models.random_forest import Random_Forest
from Models.xgboost import XG_Boost
from Models.lightgbm import LightGBM
from Models.decision_tree import Decision_Tree
from Models.svm import SVM
from Models.knn import KNN
from Models.simple_transformer import SimpleTransformer


data_dir = os.path.join("Data", "Dataset")
datasets = ["data_train.csv", "data_val.csv", "data_test.csv", "label_train.csv", "label_val.csv", "label_test.csv"]

models = [Logistic_Regression(data_dir, datasets),
          Random_Forest(data_dir, datasets),
          XG_Boost(data_dir, datasets),
          LightGBM(data_dir, datasets),
          Decision_Tree(data_dir, datasets),
          SVM(data_dir, datasets),
          KNN(data_dir, datasets),
          SimpleTransformer(data_dir, datasets)]

# models = [SimpleTransformer(data_dir, datasets)]

scores = list()
for model in models:
    model.train()
    model.validate()
    scores.append(model.test())

# Create a DataFrame from the scores and sort it by the best AUC/ROC score
performance_table = pd.DataFrame(scores, columns=["Classifier", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"])
# performance_table = performance_table.sort_values(by="ROC AUC", ascending=False).reset_index(drop=True)
# print(performance_table)

performance_table.to_csv(os.path.join("Results", "result.csv"))