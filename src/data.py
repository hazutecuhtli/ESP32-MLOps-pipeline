# src/data.py
# ******************************************************************************************************
# Importing Libraries
# ******************************************************************************************************
from influxdb_client import InfluxDBClient
import argparse
import pandas as pd, os
from dotenv import load_dotenv
from utils import local_tz
load_dotenv()

# ******************************************************************************************************
# Functions
# ******************************************************************************************************

# Functio retrieve data from InfluxDB
def extract_from_influx(hours_start=180, hours_stop=12, name=None):
    client = InfluxDBClient(
        url=os.getenv("INFLUX_URL"),
        token=os.getenv("INFLUX_TOKEN"),
        org=os.getenv("INFLUX_ORG")
    )
    bucket = os.getenv("INFLUX_BUCKET")

    # Defining interval for data retrieval
    lookback_h_start = 144 if (hours_start is None or hours_stop < 0) else int(hours_start)
    lookback_h_stop = 12 if (hours_stop is None or hours_stop < 0) else int(hours_stop)

    q = f"""
    from(bucket: "{bucket}")
      |> range(start: -{lookback_h_start}h, stop: -{lookback_h_stop}h)
      |> filter(fn: (r) => r._measurement == "dht22")
      |> filter(fn: (r) => r._field == "temp_c")
      |> map(fn: (r) => ({{ r with _value: float(v: r._value) }}))
      |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
      |> pivot(rowKey: ["_time","topic"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time","topic","temp_c"])
      |> sort(columns: ["_time"])
    """

    # Retreiving data
    df = client.query_api().query_data_frame(q)
    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True)

    # Closing client
    client.close()
    
    # Formatting retrieved data
    df = (df.rename(columns={"_time": "time", "topic": "location", "temp_c": "temp"})
          [["time", "location", "temp"]]
          .sort_values("time")
          .reset_index(drop=True))
    

    # Normalizing
    df["location"] = df["location"].astype(str).apply(lambda x: x.split('/')[1] if '/' in x else x)
    locations = ['LivingRoom', 'Room', 'Kitchen', 'office']
    for loc in locations:
        df.loc[df["location"][df["location"].str.contains(loc)].index, 'location'] = loc.lower()

    # Timezone local
    df["time"] = df["time"].dt.tz_convert(local_tz())

    # Define the date from which valid measurements started bien recorded correctlu
    fecha_corte = "2025-11-02 00:00:00"

    # Filter and reset index
    df = df[df["time"] >= fecha_corte].reset_index(drop=True)

    # Pivot table per location
    pivot_temp = df.pivot(index="time", columns="location", values="temp")
    pivot_temp.columns = [f"temp_{col}" for col in pivot_temp.columns]
    out = pivot_temp.sort_values("time")

    # Removing nan
    out.drop(columns=["temp_room", "temp_livingroom"], inplace=True, errors="ignore")

    if name==None:
        out.to_parquet("src/data/raw.parquet")
    else:
        out.to_parquet(f"src/data/raw_{name}.parquet")

    return out


# ******************************************************************************************************
# Main
# ******************************************************************************************************

if __name__ == "__main__":

    # Getting inputs from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours_start", type=int, default=144,
                        help="Interval for data retrieving")
    parser.add_argument("--hours_stop", type=int, default=12,
                        help="Interval for data retrieving")    
    parser.add_argument("--name", type=str, default=None,
                        help="Defiming if created data will be used for training or prediction")
    args = parser.parse_args()

    # Retrieving Data
    extract_from_influx(args.hours_start, args.hours_stop, args.name)

# ******************************************************************************************************
# Fin
# ******************************************************************************************************
