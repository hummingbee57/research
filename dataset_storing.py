"""
Stores streamlined code for cal_fire_temps dataset so it can be imported into the
Jupyter Notebooks for the models
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import RepeatedKFold

calfires = gpd.read_file("fire21_2.gdb")

calfires_trim = calfires.drop(labels=["STATE", "FIRE_NAME", "AGENCY", "UNIT_ID", "INC_NUM", "COMMENTS", "REPORT_AC", "C_METHOD", "OBJECTIVE", "FIRE_NUM", "COMPLEX_NAME", "COMPLEX_INCNUM", "CONT_DATE"], axis=1)
calfires_trim = calfires_trim.dropna()

# Calculate centroids and place them into calfires_trim
calfires_trim["centroid_x"] = None
calfires_trim["centroid_y"] = None
def get_centroid(fire: pd.Series):
    shape = fire["geometry"]
    centroid = shape.centroid.xy
    fire["centroid_x"] = centroid[0][0]
    fire["centroid_y"] = centroid[1][0]
    return fire

calfires_trim = calfires_trim.apply(lambda row: get_centroid(row), axis=1)

# Format Dates in calfires 
def date_trim(date):
    return date[:10]

calfires_trim.ALARM_DATE = calfires_trim.ALARM_DATE.map(date_trim)
calfires_trim["ALARM_DATE"] = pd.to_datetime(calfires_trim["ALARM_DATE"], format="%Y-%m-%d", errors="coerce")

# Add Temperature Columns
cal_temps_csv = open("cal_temps.csv")
cal_temps = pd.read_csv(cal_temps_csv)

cal_temps_trim = cal_temps.drop(labels=["Unnamed: 0", "AverageTemperatureUncertainty", "Country", "InCal"], axis=1)
cal_temps_trim.dropna(inplace=True)

cal_temps_trim["dt"] = pd.to_datetime(cal_temps_trim["dt"], format="%Y-%m-%d")

# Create a list of all of the temperatures in cal_temps associated with parameter date
def find_temps(date: pd.Timestamp):
    mod_date = pd.Timestamp(f"{date.year}-{date.month}-01")
    date_temps = cal_temps_trim.loc[cal_temps_trim["dt"] == mod_date]
    temps_list = []
    for i in date_temps.index:
        temps_list.append(date_temps["AverageTemperature"][i])
    
    return temps_list

# Gte list of all the first days of each of the 12 months before date
def get_dates_list(date):
    return pd.date_range(end=date, freq="M", periods=12)

# Store all of the indexes with fires that don't have a date associated with them
global null_indices
null_indices = [21138] # 21138 is a little shit that escapes notice by being "NaT" instead of None

# Get the list of dates for each number
def assign_temperatures(fire):
    date = fire["ALARM_DATE"]
    try:
        for i in range(1, 13):
            fire[f"{i}_MONTHS_BEFORE"] = find_temps(get_dates_list(date)[i - 1])
    except:
        if not i in null_indices:
                null_indices.append(i)
    return fire

cal_fire_temps = calfires_trim.apply(lambda row: assign_temperatures(row), axis=1)

# Check for missing data
for i in cal_fire_temps.index:
    for j in range(1, 13):
        if cal_fire_temps[f"{j}_MONTHS_BEFORE"][i] == [] or (not type(cal_fire_temps[f"{j}_MONTHS_BEFORE"][i]) == list) and not i in null_indices:
                null_indices.append(i)

    if not type(cal_fire_temps["Shape_Area"][i]) == np.float64 and not i in null_indices:
        null_indices.append(i)

cal_fire_temps = cal_fire_temps.drop(null_indices)


print("this is working")

# Prepare Data for training
fire_X = cal_fire_temps.loc[cal_fire_temps.index, ["1_MONTHS_BEFORE", "2_MONTHS_BEFORE", "3_MONTHS_BEFORE", "4_MONTHS_BEFORE", "5_MONTHS_BEFORE", "6_MONTHS_BEFORE", "7_MONTHS_BEFORE", "8_MONTHS_BEFORE", "9_MONTHS_BEFORE", "10_MONTHS_BEFORE", "11_MONTHS_BEFORE", "12_MONTHS_BEFORE"]]
fire_centroid_x = cal_fire_temps["centroid_x"]
fire_centroid_y = cal_fire_temps["centroid_y"]
fire_area = cal_fire_temps["Shape_Area"]

df_list = []
for i in range(1, 13): 
    df_list.append(pd.DataFrame(fire_X[f"{i}_MONTHS_BEFORE"].values.tolist()))

fire_X = pd.concat(df_list, axis=1)

global null_indices_2
null_indices_2 = []

for i in fire_X.index:
    for column in fire_X.columns:
         if not type(fire_X.iloc[i, column]) == np.float64 and not i in null_indices_2:
              print(i)
              null_indices_2.append(i)

fire_X = fire_X.drop(null_indices_2)
fire_area = fire_area.drop(null_indices_2)
fire_centroid_x = fire_centroid_x.drop(null_indices_2)
fire_centroid_y = fire_centroid_y.drop(null_indices_2)


k_fold = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)