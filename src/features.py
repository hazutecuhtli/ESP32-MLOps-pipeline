# ******************************************************************************************************
# Importing Libraries
# ******************************************************************************************************
# src/features.py
import pandas as pd
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="is_datetime64tz_dtype is deprecated")
# ******************************************************************************************************
# Fucntions
# ******************************************************************************************************

# Function to create the slinding window for feature generation
def create_sliding_windows(series, window_size=4):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    cols = [f"lag_{window_size-i}" for i in range(window_size)]
    return np.array(X), np.array(y), cols

# Generating the model input features
def make_features(name=None):

    # Retrieving data for feature generation
    if name == None:
        df = pd.read_parquet("src/data/raw.parquet").sort_values("time").reset_index(drop=False)
    else:
        df = pd.read_parquet("src/data/raw_pred.parquet").sort_values("time").reset_index(drop=False)

    # Data imputation
    df.interpolate(method='linear', limit_direction='both', inplace=True)

    # Assuring correct time zone
    if not pd.api.types.is_datetime64tz_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], utc=True)    

    # Hourly resampling
    df_temp = df.copy(deep=True)
    df_temp['_time'] = pd.to_datetime(df_temp['time'])
    df_temp = df_temp.sort_values('_time').set_index('_time')
    df_hourly = df_temp.resample('h').mean(numeric_only=True)

    # Reviewing dataframe structure
    needed = ['temp_kitchen', 'temp_office']
    missing = [c for c in needed if c not in df_hourly.columns]
    if missing:
        raise ValueError(f"Missing columns in data/raw.parquet: {missing}")

    # Filtering out unnecessary columns
    df_hourly = df_hourly[needed]
    df_hourly.interpolate(method='linear', limit_direction='both', inplace=True)

    # Sliding windows
    window_size = 4
    series_kit = df_hourly["temp_kitchen"].values
    series_off = df_hourly["temp_office"].values
    X_kitchen, y_kitchen, cols_kit = create_sliding_windows(series_kit, window_size)
    X_office,  y_office,  cols_off = create_sliding_windows(series_off, window_size)

    # Saving generated features
    if name==None:
        filename_kit = "src/data/feat_kit.parquet"
        filename_off = "src/data/feat_off.parquet"
    else:
        filename_kit = f"src/data/feat_kit_{name}.parquet"
        filename_off = f"src/data/feat_off_{name}.parquet"
    pd.DataFrame(
        data=np.concatenate((X_kitchen, y_kitchen.reshape(-1, 1)), axis=1),
        columns=cols_kit + ['target']
    ).to_parquet(filename_kit, index=False)

    pd.DataFrame(
        data=np.concatenate((X_office, y_office.reshape(-1, 1)), axis=1),
        columns=cols_off + ['target']
    ).to_parquet(filename_off, index=False)


# ******************************************************************************************************
# Main
# ******************************************************************************************************

if __name__ == "__main__":

    # Getting arguments from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None,
                        help="Defiming if created data will be used for training or prediction")
    args = parser.parse_args()

    # Generating features
    make_features(args.name)

# ******************************************************************************************************
# Fin
# ******************************************************************************************************
