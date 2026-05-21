import yfinance as yf
import pandas as pd

def download_data(
    ticker="AAPL",
    start="2020-01-01"
):

    data = yf.download(
        ticker,
        start=start
    )

    data.to_csv(
        f"data/raw/{ticker}.csv"
    )

    return data