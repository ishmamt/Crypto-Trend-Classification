import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from fetch_data import fetch_crypto_data


def add_target_column(data, target_col="Target"):
    """
    Adds a target column to the input DataFrame, indicating whether the previous day's closing price was up or down.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing stock data.
    target_col (str): The name of the target column to be added (default is "Target").

    Returns:
    data (pd.DataFrame): The DataFrame with the new target column added.
    """

    # Assign "1" (UP) or "0" (DOWN) based on the sign of the price difference between consecutive closing prices
    # TO-DO: Change this using sklearn label encoder for more flexibility
    data[target_col] = data["Close"].diff().apply(lambda x: "1" if x > 0 else "0")
    
    return data


def preprocess_data(data, categorical_cols=None, numerical_cols=None, target_col="Target"):
    """
    Encodes categorical variables and standardizes/normalizes numerical variables in the input DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing stock data.
    categorical_cols (list): A list of categorical column names to be encoded (default is None).
    numerical_cols (list): A list of numerical column names to be standardized/normalized (default is None).
    target_col (str): The name of the target column (default is "Target").

    Returns:
    data (pd.DataFrame): The preprocessed DataFrame.
    """

    # Make a copy of the input DataFrame to avoid modifying the original data
    data = data.copy()

    # Encode categorical columns
    if categorical_cols:
        le = LabelEncoder()
        for col in categorical_cols:
            data[col] = le.fit_transform(data[col])

    # Standardize/normalize numerical columns
    if numerical_cols:
        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Ensure the target column is the last column in the DataFrame
    data = data[[col for col in data.columns if col != target_col] + [target_col]]

    return data


def split_dataset(data, target_col="Target", test_size=0.2, exclude_columns=None):
    """
    Splits the dataset into train and test splits.

    Parameters:
    data (pd.DataFrame): The preprocessed DataFrame containing stock data.
    target_col (str): The name of the target column (default is "Target").
    test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
    exclude_columns (list): A list of column names to be excluded from the features (default is None).

    Returns:
    X_train (pd.DataFrame): A DataFrame containing the training data split.
    X_test (pd.DataFrame): A DataFrame containing the testing data split.
    y_train (pd.Series): A Series containing the labels for the training data split.
    y_test (pd.Series): A Series containing the labels for the testing data split.
    """

    # Exclude the specified columns from the features, if provided
    if exclude_columns:
        feature_columns = [col for col in data.columns if col not in exclude_columns + [target_col]]
    else:
        feature_columns = [col for col in data.columns if col != target_col]

    # Split the data into training and testing sets
    X = data[feature_columns]
    y = data[target_col]

    # TimeSeriesSplit is used to preserve the continuity of time series data
    tss = TimeSeriesSplit(n_splits=2, test_size=int(len(X.index) * test_size))
    
    for train_index, test_index in tss.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, X_test, y_train, y_test


def create_crypto_dataset(save_dir="Dataset/", ticker="BTC-USD", start="2009-01-01", end="2023-12-01"):
    """
    Creates crypto dataset for the specified ticker and segments it into train and test sets.

    Parameters:
    save_dir (str): The directory where the datasets will be stored. (default is "./Dataset/")
    tickers (str): Name of crypto ticker (default is "BTC-USD").
    start (str): The start date for fetching data in the format YYYY-MM-DD (default is "2010-01-01").
    end (str): The end date for fetching data in the format YYYY-MM-DD (default is "2022-12-31").
    """

    data = fetch_crypto_data(ticker=ticker, start=start, end=end)

    # Add a target column for prediction
    data = add_target_column(data)

    # Pre-process the data by normalizing and encoding features
    data = preprocess_data(data, numerical_cols=["Open", "High", "Low", "Close", "Adj Close", "Volume"])

    # Divide the dataset into train-test splits
    X_train, X_test, y_train, y_test = split_dataset(data, exclude_columns=['Date'])

    # Check if the save_dir exists. If not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the created splits as csv files
    X_train.to_csv(os.path.join(save_dir, "data_train.csv"), index=False)
    X_test.to_csv(os.path.join(save_dir, "data_test.csv"), index=False)
    y_train.to_csv(os.path.join(save_dir, "label_train.csv"), index=False)
    y_test.to_csv(os.path.join(save_dir, "label_test.csv"), index=False)


if __name__ == "__main__":
    create_crypto_dataset()