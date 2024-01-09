# Data Explanation
## Raw Data
-  [fire21_2.gdb](raw-data/fire21_2.gdb/) - GeoDatabase of fires downloaded from the [CAL FIRE website](https://web.archive.org/web/20230527011914/https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program/fire-perimeters)
- [cal_fire_temps.csv](raw-data/cal_fire_temps.csv) - Combined fire and temperature data - used to train the models 
- 

## Data
*  {control/mlp/mlr}_{area/cent_x/cent_y}.csv - metrics from each iteration of the model listed in the title of the file
    - trial: trial #, out of 25
    - mean_error_train: Mean absolute error calculated over training data
    - mean_error test: Mean absolute error calculated over testing data
    - r2_score_train: R^2^ scores calculated over training data
    - r2_score_test: R^2^ scores calculated over testing data
* [anova_p_values.csv](anova_p_values.csv) - p values from a one-way ANOVA test run on the metrics collected
* [f_scores.csv](f_scores.csv) - f scores from the ANOVA test