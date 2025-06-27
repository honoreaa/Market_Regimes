#load data
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df