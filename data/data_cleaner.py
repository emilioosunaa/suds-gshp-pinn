import pandas as pd
import os
import glob

raw_data_path = "data/raw"
output_path = "data/processed"

def clean_data(raw_data_path="data/raw", output_path="data/processed"):
    df_list = []
    for file_path in glob.glob(os.path.join(raw_data_path, "*.csv")):
        if "Drainage" in file_path:
            continue
        filename = os.path.basename(file_path)
        df = process_data(file_path)
        df_list.append(df)
    
    # Merge all dataframes into one
    df_merged = pd.concat(df_list, axis=1)
    df_merged = df_merged.resample('15min').interpolate()
    df_merged = df_merged.drop(columns="Suction_Z550") # this column is empty
    # Remove duplicates while keeping the first occurrence
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
    df_merged.to_csv(os.path.join(output_path, "merged_data.csv"))


def process_data(filename):
    df = pd.read_csv(filename, parse_dates=["Time"])
    df.set_index("Time", inplace=True)
    
    # Rename columns based that are repeated in datasets
    if "SoilTemp" in filename:
        df.columns = ["Soil_T_" + col if col.startswith("Z") else col for col in df.columns]
    elif "VWC_Filtered" in filename:
        df.columns = ["VWC_" + col if col.startswith("Z") else col for col in df.columns]
    elif "Suction_Filtered" in filename:
        df.columns = ["Suction_" + col if col.startswith("Z") else col for col in df.columns]
    
    # Resampling
    if "24h" in filename:
        df = df.resample('15min').ffill()
    elif "3h" in filename:
        df.index = df.index.floor('3h')
        df = df.resample('15min').ffill()
    
    return df

if __name__ == "__main__":
    clean_data(raw_data_path, output_path)
