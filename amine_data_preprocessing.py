import pandas as pd
pd.set_option("display.max_rows", 5)
pd.set_option("display.max_columns", None)
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt
import keras_tuner as kt
import timeit
import time
import io
from contextlib import redirect_stdout, redirect_stderr
import gopython

# function to randomly drop a specified number of non-pure points
def drop_random_non_pure_points(df, num_to_drop):
    if num_to_drop >= len(df):
        return pd.DataFrame()  # Return an empty DataFrame if trying to drop more rows than available
    drop_indices = np.random.choice(df.index, num_to_drop, replace=False)
    return df.drop(drop_indices)

phaseEquilibriumData = pd.read_csv("./small_sample.csv")
print(phaseEquilibriumData)

#featureNames = ['Temperature', 'Pressure', 'z_EC', 'z_DMC', 'z_Tot']
#targetNames = ['x_DMC', 'x_EC', 'x_Tot', 'y_EC', 'y_DMC', 'y_Tot']
moleFractionNames = ['z_water', 'z_ABSORBENT', 'z_CO2', 'z_N2']
mWNames = ['mW_water','mW_ABSORBENT','mW_CO2','mW_N2']
mW = [18.015,61.08,44.01, 28.01]

# drop columns where all values are 0
phaseEquilibriumData = phaseEquilibriumData.dropna()

phaseEquilibriumData['isPureLiquid'] = phaseEquilibriumData.apply(lambda row:
                                                               (row['y_water'] == 0) and
                                                               (row['y_ABSORBENT'] == 0) and
                                                               (row['y_CO2'] == 0) and
                                                               (row['y_N2'] == 0),
                                                               axis=1).astype(int)

phaseEquilibriumData['isPureVapour'] = phaseEquilibriumData.apply(lambda row:
                                                               (row['x_water'] == 0) and
                                                               (row['x_ABSORBENT'] == 0) and
                                                               (row['x_CO2'] == 0) and
                                                               (row['x_N2'] == 0),
                                                               axis=1).astype(int)

phaseEquilibriumData['inverseTemperature'] = 1/phaseEquilibriumData['Temperature']

# why not turn into mass ratios instead?
phaseEquilibriumData[mWNames] = phaseEquilibriumData[moleFractionNames]*mW

print('Printing dfs now')
print(phaseEquilibriumData.describe())

print(phaseEquilibriumData[mWNames][:5])
print(phaseEquilibriumData[moleFractionNames][:5])

print(np.sum((phaseEquilibriumData[mWNames][:5]), axis=1))

print(np.sum((phaseEquilibriumData[moleFractionNames][:5]), axis=1))

print(len(phaseEquilibriumData[phaseEquilibriumData['isPureLiquid'] == 1])*100/(len(phaseEquilibriumData[phaseEquilibriumData['isPureLiquid'] == 1])+len(phaseEquilibriumData[phaseEquilibriumData['isPureLiquid'] == 0])), '%')

print(len(phaseEquilibriumData[phaseEquilibriumData['isPureVapour'] == 1])*100/(len(phaseEquilibriumData[phaseEquilibriumData['isPureVapour'] == 1])+len(phaseEquilibriumData[phaseEquilibriumData['isPureVapour'] == 0])), '%')

featureNames = ['inverseTemperature', 'Pressure', 'z_water', 'z_ABSORBENT', 'z_CO2', 'z_N2','mW_water','mW_ABSORBENT','mW_CO2','mW_N2']
# featureNames = ['inverseTemperature', 'Pressure', 'z_water', 'z_ABSORBENT', 'z_CO2', 'z_N2']
targetNamesRegression = ['x_ABSORBENT', 'x_CO2','x_N2', 'y_water', 'y_ABSORBENT', 'y_CO2']
targetNamesClassification = ['isPureVapour', 'VapourPhaseFraction']

# Filter rows where both 'all_y_zero' and 'all_x_zero' are 0
phaseEquilibriumDataRegression = phaseEquilibriumData[(phaseEquilibriumData['isPureLiquid'] == 0) & (phaseEquilibriumData['isPureVapour'] == 0)]


# Filter for non-pure points
nonPurePoints = phaseEquilibriumData[(phaseEquilibriumData['isPureLiquid'] == 0) & (phaseEquilibriumData['isPureVapour'] == 0)]

# Function to randomly drop a specified number of non-pure points
def drop_random_non_pure_points(df, num_to_drop):
    if num_to_drop >= len(df):
        return pd.DataFrame()  # Return an empty DataFrame if trying to drop more rows than available
    drop_indices = np.random.choice(df.index, num_to_drop, replace=False)
    return df.drop(drop_indices)

# Specify the number of points to drop
num_to_drop = int(len(nonPurePoints)*0.5)  # Change this to your desired number

# Drop random non-pure points
phaseEquilibriumDataReduced = drop_random_non_pure_points(nonPurePoints, num_to_drop)

# First, identify the pure liquid and vapor points
pureLiquidPoints = phaseEquilibriumData[phaseEquilibriumData['isPureLiquid'] == 1]
pureVaporPoints = phaseEquilibriumData[phaseEquilibriumData['isPureVapour'] == 1]

# Assuming phaseEquilibriumDataReduced is the reduced set of non-pure points from the previous step
# Concatenate the pure points with the reduced non-pure points to form the classification dataset
phaseEquilibriumDataClassification = pd.concat([pureLiquidPoints, pureVaporPoints, phaseEquilibriumDataReduced])

# Reset the index of the new DataFrame, if desired
phaseEquilibriumDataClassification.reset_index(drop=True, inplace=True)

# phaseEquilibriumData = phaseEquilibriumData.loc[:, (phaseEquilibriumData != 0).any(axis=0)]
# # print(phaseEquilibriumData.shape)
# phaseEquilibriumData = phaseEquilibriumData = phaseEquilibriumData[
#     (phaseEquilibriumData['x_ABSORBENT'] != 0) |
#     ((phaseEquilibriumData['x_CO2'] != 0) | (phaseEquilibriumData['x_N2'] != 0))
# ]
# phaseEquilibriumData = phaseEquilibriumData = phaseEquilibriumData[
#     (phaseEquilibriumData['y_ABSORBENT'] != 0) |
#     ((phaseEquilibriumData['y_CO2'] != 0) | (phaseEquilibriumData['y_N2'] != 0))
# ]


# xNotNullRows = phaseEquilibriumData[(phaseEquilibriumData['x_EC'] == 0) | (phaseEquilibriumData['x_DMC'] == 0)]
# yNotNullRows = phaseEquilibriumData[(phaseEquilibriumData['y_EC'] == 0) | (phaseEquilibriumData['y_DMC'] == 0)]
# print(xNotNullRows.shape[0], yNotNullRows.shape[0])
#
# print(xNotNullRows.head())
# print(yNotNullRows.head())

phaseEquilibriumData = phaseEquilibriumData.sort_values(by=['inverseTemperature'], ascending=[True])
phaseEquilibriumDataClassification = phaseEquilibriumDataClassification.sort_values(by=['inverseTemperature'], ascending=[True])
phaseEquilibriumDataRegression = phaseEquilibriumDataRegression.sort_values(by=['inverseTemperature'], ascending=[True])

# print(phaseEquilibriumData.Temperature.mean(), phaseEquilibriumData.Temperature.min(), phaseEquilibriumData.Temperature.max())
# print(phaseEquilibriumData.Pressure.mean(), phaseEquilibriumData.Pressure.min(), phaseEquilibriumData.Pressure.max())


# Define subplots for temperature
# fig_temp, axs_temp = plt.subplots(2, 2, figsize=(12, 10))
# fig_temp.suptitle('Variables plotted against Temperature')
#
# # x_DMC vs Temperature
# axs_temp[0, 0].scatter(phaseEquilibriumData['Temperature'], phaseEquilibriumData['x_DMC'])
# axs_temp[0, 0].set_title('x_DMC vs Temperature')
# axs_temp[0, 0].set_xlabel('Temperature')
# axs_temp[0, 0].set_ylabel('x_DMC')
#
# # x_EC vs Temperature
# axs_temp[0, 1].scatter(phaseEquilibriumData['Temperature'], phaseEquilibriumData['x_EC'])
# axs_temp[0, 1].set_title('x_EC vs Temperature')
# axs_temp[0, 1].set_xlabel('Temperature')
# axs_temp[0, 1].set_ylabel('x_EC')
#
# # y_DMC vs Temperature
# axs_temp[1, 0].scatter(phaseEquilibriumData['Temperature'], phaseEquilibriumData['y_DMC'])
# axs_temp[1, 0].set_title('y_DMC vs Temperature')
# axs_temp[1, 0].set_xlabel('Temperature')
# axs_temp[1, 0].set_ylabel('y_DMC')
#
# # y_EC vs Temperature
# axs_temp[1, 1].scatter(phaseEquilibriumData['Temperature'], phaseEquilibriumData['y_EC'])
# axs_temp[1, 1].set_title('y_EC vs Temperature')
# axs_temp[1, 1].set_xlabel('Temperature')
# axs_temp[1, 1].set_ylabel('y_EC')
#
# plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust subplots to fit the figure area.
#
# # Define subplots for pressure
# fig_press, axs_press = plt.subplots(2, 2, figsize=(12, 10))
# fig_press.suptitle('Variables plotted against Pressure')
#
# # x_DMC vs Pressure
# axs_press[0, 0].scatter(phaseEquilibriumData['Pressure'], phaseEquilibriumData['x_DMC'])
# axs_press[0, 0].set_title('x_DMC vs Pressure')
# axs_press[0, 0].set_xlabel('Pressure')
# axs_press[0, 0].set_ylabel('x_DMC')
#
# # x_EC vs Pressure
# axs_press[0, 1].scatter(phaseEquilibriumData['Pressure'], phaseEquilibriumData['x_EC'])
# axs_press[0, 1].set_title('x_EC vs Pressure')
# axs_press[0, 1].set_xlabel('Pressure')
# axs_press[0, 1].set_ylabel('x_EC')
#
# # y_DMC vs Pressure
# axs_press[1, 0].scatter(phaseEquilibriumData['Pressure'], phaseEquilibriumData['y_DMC'])
# axs_press[1, 0].set_title('y_DMC vs Pressure')
# axs_press[1, 0].set_xlabel('Pressure')
# axs_press[1, 0].set_ylabel('y_DMC')
#
# # y_EC vs Pressure
# axs_press[1, 1].scatter(phaseEquilibriumData['Pressure'], phaseEquilibriumData['y_EC'])
# axs_press[1, 1].set_title('y_EC vs Pressure')
# axs_press[1, 1].set_xlabel('Pressure')
# axs_press[1, 1].set_ylabel('y_EC')
#
# plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust subplots to fit the figure area.

# Display the plots
# plt.show()

X = phaseEquilibriumData[featureNames]
y = phaseEquilibriumData[targetNamesClassification]

X_train_classification_full, X_test_classification_full, y_train_classification_full, y_test_classification_full = train_test_split(X, y, test_size=0.2, random_state=42)

vapourPhaseFraction = 1.0

# Filter for non-pure points
nonPurePoints_X_train = X_train_classification_full[(y_train_classification_full['VapourPhaseFraction'] < vapourPhaseFraction)]
nonPurePoints_X_test = X_test_classification_full[(y_test_classification_full['VapourPhaseFraction'] < vapourPhaseFraction)]
nonPurePoints_y_train = y_train_classification_full[(y_train_classification_full['VapourPhaseFraction'] < vapourPhaseFraction)]
nonPurePoints_y_test = y_test_classification_full[(y_test_classification_full['VapourPhaseFraction'] < vapourPhaseFraction)]

percentageOfPointsToDrop = 0.0
# Drop random non-pure points
X_train_reduced = drop_random_non_pure_points(nonPurePoints_X_train, int(len(nonPurePoints_X_train)*percentageOfPointsToDrop))
X_test_reduced = drop_random_non_pure_points(nonPurePoints_X_test, int(len(nonPurePoints_X_test)*percentageOfPointsToDrop))
y_train_reduced = drop_random_non_pure_points(nonPurePoints_y_train, int(len(nonPurePoints_y_train)*percentageOfPointsToDrop))
y_test_reduced = drop_random_non_pure_points(nonPurePoints_y_test, int(len(nonPurePoints_y_test)*percentageOfPointsToDrop))


print(X_train_reduced.shape[0]/nonPurePoints_X_train.shape[0])
print(X_test_reduced.shape[0]/nonPurePoints_X_test.shape[0])
print(y_train_reduced.shape[0]/nonPurePoints_y_train.shape[0])
print(y_test_reduced.shape[0]/nonPurePoints_y_test.shape[0])
# First, identify the pure liquid and vapor points

pureVaporPoints_X_train = X_train_classification_full[(y_train_classification_full['VapourPhaseFraction'] >= vapourPhaseFraction)]
pureVaporPoints_X_test = X_test_classification_full[(y_test_classification_full['VapourPhaseFraction'] >= vapourPhaseFraction)]
pureVaporPoints_y_train = y_train_classification_full[(y_train_classification_full['VapourPhaseFraction'] >= vapourPhaseFraction)]
pureVaporPoints_y_test = y_test_classification_full[(y_test_classification_full['VapourPhaseFraction'] >= vapourPhaseFraction)]


# Assuming phaseEquilibriumDataReduced is the reduced set of non-pure points from the previous step
# Concatenate the pure points with the reduced non-pure points to form the classification dataset
X_train_classification = pd.concat([pureVaporPoints_X_train, X_train_reduced])
X_test_classification = pd.concat([pureVaporPoints_X_test, X_test_reduced])
y_train_classification = pd.concat([pureVaporPoints_y_train, y_train_reduced])
y_test_classification = pd.concat([pureVaporPoints_y_test, y_test_reduced])
print(y_train_classification_full.columns)
y_train_classification_full = y_train_classification_full.drop('VapourPhaseFraction', axis=1)
y_test_classification_full = y_test_classification_full.drop('VapourPhaseFraction', axis=1)
print(y_train_classification_full.columns)
X_train_classification_full.to_csv('./trainingData/X_train_classification_full.csv', index=False)
y_train_classification_full.to_csv('./trainingData/y_train_classification_full.csv', index=False)
X_test_classification_full.to_csv('./trainingData/X_test_classification_full.csv', index=False)
y_test_classification_full.to_csv('./trainingData/y_test_classification_full.csv', index=False)

y_train_classification = y_train_classification.drop('VapourPhaseFraction', axis=1)
y_test_classification = y_test_classification.drop('VapourPhaseFraction', axis=1)

X_train_classification.to_csv('./trainingData/X_train_classification.csv', index=False)
y_train_classification.to_csv('./trainingData/y_train_classification.csv', index=False)
X_test_classification.to_csv('./trainingData/X_test_classification.csv', index=False)
y_test_classification.to_csv('./trainingData/y_test_classification.csv', index=False)

X = phaseEquilibriumData[featureNames]
y = phaseEquilibriumData[targetNamesRegression]

X_train_regression_full, X_test_regression_full, y_train_regression_full, y_test_regression_full = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_regression_full.to_csv('./trainingData/X_train_regression_full.csv', index=False)
y_train_regression_full.to_csv('./trainingData/y_train_regression_full.csv', index=False)
X_test_regression_full.to_csv('./trainingData/X_test_regression_full.csv', index=False)
y_test_regression_full.to_csv('./trainingData/y_test_regression_full.csv', index=False)


X_train_regression = X_train_regression_full[(y_train_classification_full['isPureVapour'] == 0)]
X_test_regression = X_test_regression_full[(y_test_classification_full['isPureVapour'] == 0)]
y_train_regression = y_train_regression_full[(y_train_classification_full['isPureVapour'] == 0)]
y_test_regression = y_test_regression_full[(y_test_classification_full['isPureVapour'] == 0)]

X_train_regression.to_csv('./trainingData/X_train_regression.csv', index=False)
y_train_regression.to_csv('./trainingData/y_train_regression.csv', index=False)
X_test_regression.to_csv('./trainingData/X_test_regression.csv', index=False)
y_test_regression.to_csv('./trainingData/y_test_regression.csv', index=False)

