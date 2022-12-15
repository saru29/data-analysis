# resources: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
import os
import dtreeviz as dtreeviz
import pandas as pd
from pandas import read_csv
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree


# Sow plot and clear settings
def plotshow(t):
    showPlot = True
    if t & showPlot:
        plt.show()
    plt.clf()
    plt.cla()
    plt.close()


# SELECT dataset
dataset_user1 = "user1_short.csv"
dataset_user2 = "user2.csv"
dataset_S131 = "s_131.csv"
actualDataset = ""

userSelect = int(input("Opps, try again\nUser 1 (1) data, User 2 (2) or Session 131 (3) data? "))
while userSelect < 1 and userSelect > 2:
    userSelect = int(input("Opps, try again\nUser 1 (1) data, User 2 (2) or Session 131 (3) data? "))

if userSelect == 1:
    actualDataset = dataset_user1
if userSelect == 2:
    actualDataset = dataset_user2
if userSelect == 3:
    actualDataset = dataset_S131
else:
    actualDataset = dataset_user1
# Load dataset
fDir = os.getcwd() + "/userdata"
df_train = read_csv(os.path.join(fDir, actualDataset), index_col='step')

# Dataset dynamics
print("Shape of test data : ", df_train.shape)
print("Test data Info")
print("-" * 75)
print(df_train.info())
print(df_train.head())
print(df_train.columns)

# train data (X)
X = df_train.iloc[:, :-1]
# print(X.head())

# train data (y) - Target is "power" or watts per kilogram
y = df_train.iloc[:, -1]
# print(y.head())

# Feature Selection
selection = ExtraTreesRegressor()
selection.fit(X, y)

# looking at important features given by ExtraTreesRegressor
# print(selection.feature_importances_)
plt.figure(figsize=(18, 18))
sns.heatmap(df_train.corr(), annot=True, cmap='RdYlGn')
plotshow(True)

# plot graph of feature importances for better visualisation
plt.figure(figsize=(12, 8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plotshow(True)

#random number/seed
randomNum = random.randint(1, 150)

# Fitting model using Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_test)

# Accuracy for training dataset
print("Test and Train Performance")
print("Accuracy for training dataset: ", reg_rf.score(X_train, y_train))
# Accuracy for test dataset
print("Accuracy for test dataset: ", reg_rf.score(X_test, y_test))

# Re-initialise after split
X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]

# Test entire data set
X_prod, y_prod = X, y
y_pred = reg_rf.predict(X)

# Accuracy for training dataset
print("Accuracy for production: ", reg_rf.score(X_prod, y_prod))

# -----------------------------------
# -----------------------------------
# Collect data (not required)
# X_test.to_csv("X_test.csv")
# X_train.to_csv("X_train.csv")
# y_test.to_csv("y_test.csv")
# y_train.to_csv("y_train.csv")

# Store Predictions
fDir = os.getcwd() + "/predictions"
predictions_df = pd.DataFrame(y_pred, columns=["predicted"])
predictions_df.to_csv(os.path.join(fDir, "predictions.csv"))

# DF for Results
df_results = pd.DataFrame(columns=["step", "actual", "predicted"])
indexValue = []
for x in range(0, len(y_prod), 1):
    # print output
    # print("Step: ",int(y_test.index[x]),"\t\tActual: ", y_test.iloc[x], "\t Predicted: ",y_pred[x])

    # Store Results
    df_results.loc[x] = [y_prod.index[x], y_prod.iloc[x], y_pred[x]]
    indexValue.append(x)

# Results to CSV
fDir = os.getcwd() + "/results"
df_results.to_csv(os.path.join(fDir, "results.csv"))

# Plot in Time Series:
plt.subplot(111)
power_att = df_results["actual"].to_numpy()
plt.plot(indexValue, power_att, linewidth=1.75)
plt.subplot(111)
predictions_att = predictions_df["predicted"].to_numpy()
plt.plot(indexValue, predictions_att, linewidth=1.75)
plotshow(True)
plt.close()

# Distribution Plot (Check model performance)
sns.displot(y_prod - y_pred)
plotshow(True)

# plotting the scatter plot
plt.scatter(y_prod, y_pred, alpha=0.5)
plt.plot([0, max(max(y_prod), max(y_pred))], [0, max(max(y_prod), max(y_pred))], ls="--", c=".3")
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.grid(True)
plotshow(True)

# Print final performance
print('MAE:', metrics.mean_absolute_error(y_prod, y_pred))
print('MSE:', metrics.mean_squared_error(y_prod, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_prod, y_pred)))
# r squared error
print("R2:", metrics.r2_score(y_prod, y_pred))
stop = input("Any key to continue to Hyperparameter Tuning: ")

## Hyperparameter Tuning
## Randomized Search CV
## There are two techniques of Hyperparameter tuning i.e
## 1) RandomizedSearchCv 2) GridSearchCV
## We use RandomizedSearchCv because it is much faster than GridSearchCV
## Number of trees in random forest

# Split Train and Test Set Again:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare for Hyperparameter Tuning
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# create random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# Random search of parameters, using 5 fold cross validation,
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator=reg_rf, param_distributions=random_grid, scoring='neg_mean_squared_error',
                               n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)

# Fitting model using Random Forest
rf_random.fit(X_train, y_train)

# Looking at the best parameters
rf_random.best_params_

# Re-initialise
X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]

# Test entire data set
X_prod, y_prod = X, y

# Prediction variable
prediction = rf_random.predict(X)

# Save predictions
fDir = os.getcwd() + "/predictions"
predictions_df = pd.DataFrame(prediction, columns=["predicted"])
predictions_df.to_csv(os.path.join(fDir, "predictions_hyper.csv"))

# DF for Results
df_results = pd.DataFrame(columns=["step", "actual", "predicted"])
indexValue = []
for x in range(0, len(y_prod), 1):
    # print output
    # print("Step: ",int(y_test.index[x]),"\t\tActual: ", y_test.iloc[x], "\t Predicted: ",y_pred[x])
    df_results.loc[x] = [y_prod.index[x], y_prod.iloc[x], prediction[x]]
    indexValue.append(x)

# Results to CSV
fDir = os.getcwd() + "/results"
df_results.to_csv(os.path.join(fDir, "results_hyper.csv"))

# Results:
# Plot in Time Series:
plt.subplot(111)
power_att = df_results["actual"].to_numpy()
plt.plot(indexValue, power_att, linewidth=1.75)
plt.subplot(111)
predictions_att = predictions_df["predicted"].to_numpy()
plt.plot(indexValue, predictions_att, linewidth=1.75)
plotshow(True)

# <><><><><><><><><><><><>
plt.figure(figsize=(8, 8))
sns.displot(y_prod - prediction)
plotshow(True)

# plot
plt.subplot(111)
plt.scatter(y_prod, prediction, alpha=0.5)
plt.subplot(111)
plt.plot([0, max(max(y_prod), max(y_pred))], [0, max(max(y_prod), max(y_pred))], ls="--", c=".3")
plt.xlabel("y_test")
plt.ylabel("y_pred")
plotshow(True)

# print performance
print('MAE:', metrics.mean_absolute_error(y_prod, prediction))
print('MSE:', metrics.mean_squared_error(y_prod, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_prod, prediction)))
print("R^2: ", metrics.r2_score(y_prod, prediction))
# NEXT: Use model to predict values in data.csv dataset, and plot actual vs pred. Need to track index values
