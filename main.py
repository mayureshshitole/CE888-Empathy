# -*- coding: utf-8 -*-
"""DSDM2.ipynb


Name: Mayuresh Shitole (ms22971)
Reg. No.: 2205458
Supervisor: Dr Ana Matran-Fernandez
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer,IterativeImputer,SimpleImputer
import seaborn as sns

"""**Load the Dataset**"""

# Dataset contains 60 files, so here loading one participant's .tsv file to perform our tasks
# Some files are huge and cause memory issues so we can load few percent data of that to reduce the load on machine

# num_rows = int(len(pd.read_csv('Participant0010.tsv', sep='\t')) * 0.1)
# df = pd.read_csv('Participant0010.tsv', sep='\t', nrows=num_rows)

df = pd.read_csv('Participant0010.tsv', sep='\t')

"""**Data Reading**"""

df.shape

df.head(10)

df.tail(10)

# View summary statistics of the dataset
df.describe()

df.columns

"""**Data Visualisation**"""

# bar plot for project name
plt.figure(figsize=(10,6))
sns.countplot(y='Project name', data=df)
plt.title('Project Name Count Plot')
plt.show()

# convert the 'Recording timestamp' column to datetime
df['Recording timestamp'] = pd.to_datetime(df['Recording timestamp'])

# set the 'Recording timestamp' column as the index of the DataFrame
df.set_index('Recording timestamp', inplace=True)

# plot the time series
df.plot(figsize=(10, 5))

# add labels and title to the plot
plt.xlabel('Time')
plt.ylabel('Y-axis ')
plt.title('recording timestamps')

# show the plot
plt.show()

# event count
sns.countplot(x='Event', data=df)
plt.xticks(rotation=90)
plt.show()

# bar plot for eye movement type
df['Eye movement type'].value_counts().plot.bar(figsize=(3,3))

# create scatter plot for fixation point X vs Fixation point Y
plt.scatter(df['Fixation point X'], df['Fixation point Y'])

# set plot title and axis labels
plt.title('Fixation point X vs. Fixation point Y')
plt.xlabel('Fixation point X')
plt.ylabel('Fixation point Y')

# display plot
plt.show()

# create scatter plot for Gaze point X vs Gaze point Y
plt.scatter(df['Gaze point X'], df['Gaze point Y'])
plt.xlabel('Gaze point X')
plt.ylabel('Gaze point Y')
plt.title('Scatter plot of Gaze point X vs. Gaze point Y ')
plt.show()

"""**Data Cleaning**"""

df.columns

# find columns with more than 50% null values
null_percent = df.isnull().sum()/len(df) * 100
# columns to drop
cols_to_drop = null_percent[null_percent >= 50].index

print("Columns to drop are:")
print(cols_to_drop)

#  drop the columns
df.drop(cols_to_drop, axis=1, inplace=True)

# drop the columns with single value thorughout the dataset
# loop through the columns and find the each column having a single value
for col in df.columns:
    if len(df[col].unique()) == 1:
      # drop that column
        df.drop(col,inplace=True,axis=1)

# lets see how many null values are there
df.isna().sum()

# Filling missing value. We have both numerical and categorical data, dropping null values will reduce the data but
# according to some columns I have decided to keep that and instead of dropping categorical null replacing it by "unknown" word

# Fill missing values of categorical data columns with "unknown"
object_cols = df.select_dtypes(include=['object']).columns
df[object_cols] = df[object_cols].fillna(df[object_cols].mode().iloc[0])
    
# Select only the numeric columns from the dataframe
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Impute missing values with the mean of the respective numeric columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# checking null values now
df.isna().sum()

df.hist(figsize=(30,30))

"""**Correlation Analysis** """

# Select only the numeric columns from the dataframe
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Impute missing values with the mean of the respective numeric columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())



from statsmodels.stats.outliers_influence import variance_inflation_factor

# Perform multicollinearity analysis on the float datatype columns
float_cols = df.select_dtypes(exclude=['object']).columns
X = df[float_cols].dropna()
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

X.corr()

# Plot heatmap of correlation matrix for float datatype columns
corr = X.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation matrix heatmap')

plt.show()

print(df[["Fixation point X","Gaze point right X"]].corr())
print(df[["Gaze point left X","Fixation point X"]].corr())

# Plot scatter plot
plt.scatter(df['Fixation point X'], df['Gaze point X'], label='Fixation point X')
plt.scatter(df['Gaze point right X'], df['Gaze point X'], label='Gaze point right X')
plt.xlabel('Features')
plt.ylabel('Target Variable')
plt.legend()
plt.show()

# Plot box plot
df.boxplot(column=['Gaze point X','Fixation point X', 'Gaze point right X'])
plt.show()

# here we can clean more data by dropping highly correlated columns

# find highly correlated features
highly_correlated = []
threshold = 0.7
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > threshold:
            colname = corr.columns[i]
            highly_correlated.append(colname)

print('Highly correlated columns are:')
highly_correlated

# remove highly correlated features
df = df.drop(columns=highly_correlated)

# dataframe now
df

"""**Feature Engineerin**"""

# now perform one hot encoding on this cleaned data and then again perform PCA 
# I will use LSTM and decision tree regressor to find out which one will give us good results

# import the necessary 
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

# separate categorical and numerical features
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Standardize numerical data
num_scaler = MinMaxScaler()
num_data = num_scaler.fit_transform(df[numerical_cols])

# As we have a categorical data we need to perform one hot encoding on it
# Here I am using OneHotEncoder() funciton from the sklearn preprocessing package
# One-hot encode categorical data
cat_encoder = OneHotEncoder(sparse=True)
cat_data = cat_encoder.fit_transform(df[categorical_cols])

# Combine numerical and categorical data
combined_data = np.hstack((num_data, cat_data.toarray()))

combined_data

# PCA
pca = PCA(n_components=10)
pca_data = pca.fit_transform(combined_data)

# print pca_data
pca_data

"""**LSTM**"""

# Split into train and test sets
train_size = int(len(pca_data) * 0.8)
train_data, test_data = pca_data[0:train_size,:], pca_data[train_size:len(df),:]

# Convert data to time series format

# This function creates a time series dataset where the X values are a sequence of time steps and the y value
# is the next time step after the sequence.
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        # Take a sequence of time_steps from X starting at i
        # and append it to the input sequences Xs.
        Xs.append(X[i:(i+time_steps), :])
        # Take the next value in the target dataset y and append it to the target values ys.
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10 # flexible
# declare the  train and test data
X_train, y_train = create_dataset(train_data, train_data, time_steps)
X_test, y_test = create_dataset(test_data, test_data, time_steps)

# Build LSTM model
model = Sequential()

# Add an LSTM layer with 64 units/neurons, and set the input shape to match the shape of the training data
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))

# Add a Dense layer with 64 units/neurons and relu activation function
model.add(Dense(64, activation='relu'))

# Add a Dense layer with number of units equal to the number of features in the training data
model.add(Dense(X_train.shape[2]))

# Compile the model with the Adam optimizer and mean squared error loss function
model.compile(optimizer='adam', loss='mse')

# Train the model
history=model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

# Make predictions on test set
y_pred = model.predict(X_test)

import matplotlib.pyplot as plt
# Plot training and validation loss as a function of epochs
# Plot epoch vs loss graph
plt.plot(history.history['loss'])
plt.title('LSTM Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# # Reshape y_test and y_pred to 2D arrays
# y_test = y_test.reshape(-1, pca_data.shape[1])
# y_pred = y_pred.reshape(-1, pca_data.shape[1])

# # Inverse transform the predictions and true values
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)

# Calculate MSE and MAE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print scores
print('MSE:', mse)
print('MAE:', mae)

"""Decision Tree Regressor"""

# Build Decision Tree Regressor model
tree_model = DecisionTreeRegressor()
tree_model.fit(X_train.reshape(X_train.shape[0], -1), y_train.reshape(y_train.shape[0], -1))

# Make predictions on test set
y_pred_tree = tree_model.predict(X_test.reshape(X_test.shape[0], -1))

# Calculate MSE and MAE
mse_tree = mean_squared_error(y_test, y_pred_tree)
mae_tree = mean_absolute_error(y_test, y_pred_tree)

# Print scores
print('Decision Tree MSE:', mse_tree)
print('Decision Tree MAE:', mae_tree)

# Print scores
print('MSE and MAE scores of both models we used...')
print('LSTM MSE:', mse)
print('LSTM MAE:', mae)
print('Decision Tree regressor MSE:', mse_tree)
print('Decision Tree regressor MAE:', mae_tree)





