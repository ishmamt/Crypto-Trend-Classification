import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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
    data[target_col] = data["Close"].diff().apply(lambda x: "1" if x > 0 else "0")
    
    return data


def create_crypto_dataset(save_dir="Dataset/", ticker="BTC-USD", start="2010-01-01", end="2022-12-31"):
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

    # Check if the save_dir exists. If not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data.to_csv(os.path.join(save_dir, "data.csv"), index=False)


if __name__ == "__main__":
    create_crypto_dataset()