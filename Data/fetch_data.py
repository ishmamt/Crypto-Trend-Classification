import stat
import yfinance as yf
import pandas as pd


def fetch_crypto_data(ticker="BTC-USD", start="2010-01-01", end="2022-12-31"):
    """
    Fetches crypto data for the specified ticker and time period using the yfinance library.

    Parameters:
    tickers (str): Name of crypto ticker (default is "BTC-USD").
    start (str): The start date for fetching data in the format YYYY-MM-DD (default is "2010-01-01").
    end (str): The end date for fetching data in the format YYYY-MM-DD (default is "2022-12-31").

    Returns:
    data (pd.DataFrame): A pandas DataFrame containing the fetched crypto data.
    """

    data = yf.download(ticker, start=start, end=end)

    # Reset the index and return the final DataFrame
    data.reset_index(inplace=True)
    
    return data


if __name__ == "__main__":
    data = fetch_crypto_data(start="2000-01-01", end="2023-12-01")
    print(data)