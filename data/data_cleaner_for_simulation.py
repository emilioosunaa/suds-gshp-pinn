import pandas as pd
import os
import glob
from functools import reduce

raw_data_path = "data/raw"
output_path = "data/processed"

def clean_data(raw_data_path="data/raw", output_path="data/processed"):
    df_list = []
    for file_path in glob.glob(os.path.join(raw_data_path, "*.csv")):
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        measurement = parts[1]
        if "SoilTemp" in measurement or "Suction" in measurement or "VWC" in measurement:
            df = process_data(file_path, measurement)
        elif "ThermalConductivity" in measurement:
            df = process_data(file_path, measurement, "TC")
        else:
            continue
        df.to_csv(os.path.join(output_path, measurement.lower() + ".csv"), index=False)
        df.set_index("Time", inplace=True)
        df_list.append(df)
    
    merged = reduce(lambda left, right: pd.merge(left, right, on=['Time', 'Height'], how='outer'), df_list)
    merged.sort_values(['Time','Height'], inplace=True)
    merged.to_csv(os.path.join(output_path, "merged_data_filtered.csv"))

    merged["VWC"] = merged.groupby("Time")["VWC"].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )

    merged["ThermalConductivity"] = merged.groupby("Time")["ThermalConductivity"].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
    merged["ThermalConductivity"] = merged["ThermalConductivity"].ffill().bfill()

    merged.to_csv(os.path.join(output_path, "merged_data_interpolated.csv"))

def process_data(filename, measurement, replace_char="Z"):
    df = pd.read_csv(filename, parse_dates=["Time"])

    if "3h" in filename:
        df.set_index("Time", inplace=True)
        df.index = df.index.floor('3h')
        df = df.resample('15min').interpolate()
        df.reset_index(inplace=True)

    z_columns = [col for col in df.columns if col.startswith(replace_char)]
    df_long = pd.melt(df, id_vars=['Time'], value_vars=z_columns,
                  var_name='Height', value_name=measurement)
    
    df_long['Height'] = df_long['Height'].str.replace(replace_char, '').astype(float)
    return df_long

if __name__ == "__main__":
    clean_data(raw_data_path, output_path)