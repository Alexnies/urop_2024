import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# import custom functions
from saving import save_dataset

# set pandas options
pd.set_option("display.max_rows", 5)
pd.set_option("display.max_columns", None)


# function to randomly drop a specified number of non-pure points
def drop_random_non_pure_points(df, num_to_drop):
    if num_to_drop >= len(df):
        return pd.DataFrame()  # Return an empty DataFrame if trying to drop more rows than available
    drop_indices = np.random.choice(df.index, num_to_drop, replace=False)
    return df.drop(drop_indices)


# creating DataFrame
phase_equilibrium_data = pd.read_csv("./gPROMS_data/output_6.csv")

# visualising data
attributes = ["z_water", "z_absorbent", "z_co2", "z_n2", "x_co2"]
scatter_matrix(phase_equilibrium_data[attributes], figsize=(12, 8))

# Data Preprocessing
mole_fraction_names = ['z_water', 'z_absorbent', 'z_co2', 'z_n2']

# drop rows which contains NaN
phase_equilibrium_data = phase_equilibrium_data.dropna()
# drop rows where all values are 0
phase_equilibrium_data = phase_equilibrium_data.loc[~(phase_equilibrium_data == 0).all(axis=1)]

# create new columns "isPureLiquid" and "isPureVapour" and "1/T"
phase_equilibrium_data['isPureLiquid'] = phase_equilibrium_data.apply(lambda row:
                                                                      (row['y_water'] == 0) and
                                                                      (row['y_absorbent'] == 0) and
                                                                      (row['y_co2'] == 0) and
                                                                      (row['y_n2'] == 0),
                                                                      axis=1).astype(int)

phase_equilibrium_data['isPureVapour'] = phase_equilibrium_data.apply(lambda row:
                                                                      (row['x_water'] == 0) and
                                                                      (row['x_absorbent'] == 0) and
                                                                      (row['x_co2'] == 0) and
                                                                      (row['x_n2'] == 0),
                                                                      axis=1).astype(int)

phase_equilibrium_data['inv_temp'] = 1 / phase_equilibrium_data['temp']

# checking if all mole fractions add up to 1
all_mole_frac = np.sum((phase_equilibrium_data[mole_fraction_names][:]), axis=1)
print(f"Mole fractions sum up to: {np.sum(all_mole_frac) / len(phase_equilibrium_data)}")

# calculating the percentage of pure liquid and vapour points
num_pure_liq_points = len(phase_equilibrium_data[phase_equilibrium_data['isPureLiquid'] == 1])
num_non_pure_liq_points = len(phase_equilibrium_data[phase_equilibrium_data['isPureLiquid'] == 0])
frac_pure_liq_points = num_pure_liq_points / (num_pure_liq_points + num_non_pure_liq_points)

num_pure_vap_points = len(phase_equilibrium_data[phase_equilibrium_data['isPureVapour'] == 1])
num_non_pure_vap_points = len(phase_equilibrium_data[phase_equilibrium_data['isPureVapour'] == 0])
frac_pure_vap_points = num_pure_vap_points / (num_pure_vap_points + num_non_pure_vap_points)

print(f"{frac_pure_liq_points * 100:.2f}% of the points are pure liquid.")
print(f"{(frac_pure_vap_points * 100)}% of the points are pure vapour")

phase_equilibrium_data.plot(kind="scatter", x="z_co2", y="x_co2",
                            s=phase_equilibrium_data["z_absorbent"] * 2000, label="global absorbent",
                            c="temp", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False,
                            figsize=(10, 7))

phase_equilibrium_data.plot(kind="scatter", x="z_co2", y="y_co2",
                            s=phase_equilibrium_data["z_absorbent"] * 2000, label="global absorbent",
                            c="temp", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False,
                            figsize=(10, 7))

# final data processing
features = ['inv_temp', 'pressure', 'z_water', 'z_absorbent', 'z_co2', 'z_n2']
labels = ['x_absorbent', 'x_co2', 'x_n2', 'y_water', 'y_absorbent', 'y_co2']  # x_water and y_n2 calculated from MB
classification_labels = ['isPureVapour', 'vapour_fraction']

# Filter rows where both 'all_y_zero' and 'all_x_zero' are 0
phase_equilibrium_data_regression = phase_equilibrium_data[
    (phase_equilibrium_data['isPureLiquid'] == 0) & (phase_equilibrium_data['isPureVapour'] == 0)]

# Filter for non-pure points
non_pure_points = phase_equilibrium_data[
    (phase_equilibrium_data['isPureLiquid'] == 0) & (phase_equilibrium_data['isPureVapour'] == 0)]

# Specify the number of points to drop
num_to_drop = int(len(non_pure_points) * 0.5)  # Change this to your desired number

# Drop random non-pure points
phase_equilibrium_data_reduced = drop_random_non_pure_points(non_pure_points, num_to_drop)

# First, identify the pure liquid and vapor points
pure_liquid_points = phase_equilibrium_data[phase_equilibrium_data['isPureLiquid'] == 1]
pure_vapour_points = phase_equilibrium_data[phase_equilibrium_data['isPureVapour'] == 1]

# sorting the DataFrames
phase_equilibrium_data = phase_equilibrium_data.sort_values(by=['inv_temp'], ascending=[True])
phase_equilibrium_data_regression = phase_equilibrium_data_regression.sort_values(by=['inv_temp'], ascending=[True])

# creating training and testing datasets
X = phase_equilibrium_data[features]
y = phase_equilibrium_data[labels]

save_dataset(X, y, name="regression_full")

X = phase_equilibrium_data_regression[features]
y = phase_equilibrium_data_regression[labels]

save_dataset(X, y, name="regression")
