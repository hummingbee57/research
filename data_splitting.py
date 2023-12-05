from sklearn.model_selection import RepeatedKFold
from dataset_storing import cal_fire_temps

fire_X = cal_fire_temps.loc[cal_fire_temps.index, ["ALARM_DATE", "CAUSE", "YEAR_", "1_MONTHS_BEFORE", "2_MONTHS_BEFORE", "3_MONTHS_BEFORE", "4_MONTHS_BEFORE", "5_MONTHS_BEFORE", "6_MONTHS_BEFORE", "7_MONTHS_BEFORE", "8_MONTHS_BEFORE", "9_MONTHS_BEFORE", "10_MONTHS_BEFORE", "11_MONTHS_BEFORE", "12_MONTHS_BEFORE"]]
fire_y = cal_fire_temps.loc[cal_fire_temps.index, ["geometry"]]

k_fold = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)

split_set = k_fold.split(fire_X, fire_y)

