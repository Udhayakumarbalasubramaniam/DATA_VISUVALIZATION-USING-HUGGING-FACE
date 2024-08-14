import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Handling missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Encoding categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df
