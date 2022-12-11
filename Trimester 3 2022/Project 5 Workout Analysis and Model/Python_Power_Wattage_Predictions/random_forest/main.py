# resources: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def plotshow(t):
    if t:
        plt.show()
    plt.clf()
    plt.cla()
    plt.close()


# Load dataset
# data.csv = minute intervals, 113 minutes (6,780 seconds)
# data_2.csv = second intervals
# using seconds (data_2.csv) as performance is 'better;
df_train = read_csv("data_2.csv", index_col='step')
# Dataset dynamics
print("Shape of test data : ", df_train.shape)
print("Test data Info")
print("-" * 75)
print(df_train.info())
print(df_train.head())
print(df_train.columns)

# train data (X)
X = df_train.iloc[:, :-1]
print(X.head())

# train data (y) - Target is "power" or watts per kilogram
y = df_train.iloc[:, -1]
print(y.head())

# Feature Selection
selection = ExtraTreesRegressor()
selection.fit(X, y)

# looking at important features given by ExtraTreesRegressor
print(selection.feature_importances_)

plt.figure(figsize=(18, 18))
sns.heatmap(df_train.corr(), annot=True, cmap='RdYlGn')
plotshow(True)

# plot graph of feature importances for better visualisation
plt.figure(figsize=(12, 8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plotshow(True)

# Fitting model using Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)
# y_pred--> our prediction variable for random forest regressor
y_pred = reg_rf.predict(X_test)
print("Predictions: ", y_pred)
# Accuracy for training dataset
print("Accuracy for training dataset: ", reg_rf.score(X_train, y_train))
# Accuracy for test dataset
print("Accuracy for test dataset: ", reg_rf.score(X_test, y_test))
sns.displot(y_test - y_pred)
plotshow(True)

# plotting the scatter plot
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([0, max(max(y_test), max(y_pred))], [0, max(max(y_test), max(y_pred))], ls="--", c=".3")
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.grid(True)
plotshow(True)

# print performance
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# r squared error
print("R2:", metrics.r2_score(y_test, y_pred))

## Hyperparameter Tuning
## Randomized Search CV
## There are two techniques of Hyperparameter tuning i.e
## 1) RandomizedSearchCv 2) GridSearchCV
## We use RandomizedSearchCv because it is much faster than GridSearchCV
## Number of trees in random forest
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
rf_random.fit(X_train, y_train)
# Looking at the best parameters
rf_random.best_params_

# Prediction variable
prediction = rf_random.predict(X_test)
plt.figure(figsize=(8, 8))
sns.displot(y_test - prediction)
plotshow(True)

# plot
plt.subplot(111)
plt.scatter(y_test, prediction, alpha=0.5)
plt.subplot(111)
plt.plot([0, max(max(y_test), max(y_pred))], [0, max(max(y_test), max(y_pred))], ls="--", c=".3")
plt.xlabel("y_test")
plt.ylabel("y_pred")
plotshow(True)

# print performance
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
print("R^2: ", metrics.r2_score(y_test, prediction))

# NEXT: Use model to predict values in data.csv dataset, and plot actual vs pred. Need to track index values
