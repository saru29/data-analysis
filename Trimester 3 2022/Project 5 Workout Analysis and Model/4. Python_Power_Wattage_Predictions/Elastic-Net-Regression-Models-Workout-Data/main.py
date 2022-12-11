import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from numpy import mean
from numpy import std
from numpy import absolute
from numpy import arange
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
import imageio

# Upskilling:
# resources: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
# resources: https://machinelearningmastery.com/elastic-net-regression-in-python/
def plotshow(t):
    if t:
        plt.show()
    plt.clf()
    plt.cla()
    plt.close()

#-------------------------
#Cycling Data

df = read_csv("train_test.csv", index_col='step')
#Get data and as
data = df.values
X, y = data[:, :-1], data[:, -1]
# define model
model = ElasticNet(alpha=1.0, l1_ratio=0.5)
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
print("Average Power", round(mean(df["power"].to_numpy()),2))
# xx = input("Any key to continue")

def CyclingElasticNetValues():
    # Tuning Elastic Net Hyperparameters ElasticNet
    data = df.values
    X, y = data[:, :-1], data[:, -1]
    # define model
    model = ElasticNet()
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define grid
    grid = dict()
    grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
    grid['l1_ratio'] = arange(0, 1, 0.01)
    # define search
    search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(X, y)
    # summarize
    print('MAE: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
    alpha_set = results.best_params_['alpha']
    l1_ratio_set = results.best_params_['l1_ratio']
    return results.best_params_['alpha'], results.best_params_['l1_ratio']
def CyclingElasticNetModel(alpha_set, l1_ratio_set, X, y):
    print("running!")
    model = ElasticNet(alpha=alpha_set, l1_ratio=l1_ratio_set)
    # fit model
    model.fit(X, y)
    # define new data
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    scores = absolute(scores)
    print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
    values = []
    predictions = []
    for y in range(0,len(data),1):
        for i in range(0, df.shape[1]-1, 1):
            values.append(df.iloc[y, i])
        # make a prediction
        yhat = model.predict([values])
        predictions.append(yhat)
        # # summarize prediction
        values = []

    # Store Predictions in dataframe
    predictions_df = pd.DataFrame(predictions, columns=["Predicted_Power"])
    predictions_att = predictions_df["Predicted_Power"].to_numpy()
    Power_att = df["power"].to_numpy()

    plt.subplot(111)
    plt.scatter(predictions_att, Power_att, linewidth=1.75)
    # adjust
    plt.subplots_adjust(hspace=0.558)
    plt.subplot(111)
    plt.plot([0, max(max(predictions_att),max(Power_att))], [0, max(max(predictions_att),max(Power_att))], ls="--", c=".3")
    plt.grid(True)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plotshow(True)

    # Plot in Time Series:
    plt.subplot(111)
    power_att = df["power"].to_numpy()
    plt.plot(df.index.values, power_att, linewidth=1.75)
    plt.subplot(111)
    predictions_att = predictions_df["Predicted_Power"].to_numpy()
    plt.plot(df.index.values, predictions_att, linewidth=1.75)
    plotshow(True)

    #print results to csv
    df_results = df.reindex(
        columns=["enhanced_speed", "cadence", "max_heart_rate_perct", "power", "power_prediction"])
    df_results['power_prediction'] = predictions_att
    print(df_results.head())
    df_results.to_csv("results_elastic.csv")

    X_1 = df_results[["enhanced_speed", "cadence", "max_heart_rate_perct"]]
    y_1 = df_results[["power"]]
    r2 = model.score(X_1, y_1)
    print("Model R^2: ", r2)
def CyclingElasticNetModel_V2(alpha_set, l1_ratio_set, X, y):
    print("running!")
    model = ElasticNet(alpha=alpha_set, l1_ratio=l1_ratio_set)
    # fit model
    model.fit(X, y)
    # define new data
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    scores = absolute(scores)
    print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
    values = []
    predictions = []
    for y in range(0,len(data),1):
        for i in range(0, df.shape[1]-1, 1):
            values.append(df.iloc[y, i])
        # make a prediction
        yhat = model.predict([values])
        predictions.append(yhat)
        # # summarize prediction
        values = []

    # Store Predictions in dataframe
    predictions_df = pd.DataFrame(predictions, columns=["Predicted_Power"])
    predictions_att = predictions_df["Predicted_Power"].to_numpy()
    Power_att = df["power"].to_numpy()

    #print results to csv
    df_results = df.reindex(
                columns=["enhanced_speed", "cadence", "grade", "enhanced_altitude", "max_heart_rate_perct", "power"])
    df_results['power_prediction'] = predictions_att
    print(df_results.head())
    df_results.to_csv("results_elastic_V2.csv")

    X_1 = df_results[["enhanced_speed", "cadence", "grade", "enhanced_altitude", "max_heart_rate_perct"]]
    y_1 = df_results[["power"]]
    r2 = model.score(X_1, y_1)
    print("Model R^2: ", r2)

    plt.subplot(111)
    plt.scatter(predictions_att, Power_att, linewidth=1.75)
    # adjust
    plt.subplots_adjust(hspace=0.558)
    plt.subplot(111)
    plt.plot([0, max(max(predictions_att),max(Power_att))], [0, max(max(predictions_att),max(Power_att))], ls="--", c=".3")
    plt.grid(True)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plotshow(True)

    # Plot in Time Series:
    plt.subplot(111)
    power_att = df["power"].to_numpy()
    plt.plot(df.index.values, power_att, linewidth=1.75)
    plt.subplot(111)
    predictions_att = predictions_df["Predicted_Power"].to_numpy()
    plt.plot(df.index.values, predictions_att, linewidth=1.75)
    plt.savefig('graph.png')
    plotshow(True)

#-------------------------
# Run
data = df.values
X, y = data[:, :-1], data[:, -1]
# define model
alpha_set, l1_ratio_set = CyclingElasticNetValues()
#CyclingElasticNetModel(alpha_set, l1_ratio_set, X, y)

# Different Data set: 400 rows of data, 1 second intervals
df = read_csv("data_sample_2.csv", index_col='step', nrows=400)
#Get data and as
data = df.values
X, y = data[:, :-1], data[:, -1]
#CyclingElasticNetModel(alpha_set, l1_ratio_set, X, y)

# Different Data set: 400 rows of data, 3 second intervals
df = read_csv("data_sample_3.csv", index_col='step', nrows=400)
#Get data and as
data = df.values
X, y = data[:, :-1], data[:, -1]
#CyclingElasticNetModel(alpha_set, l1_ratio_set, X, y)

# Run again though with watts per kilo and some added features - 1 minute intervals
df = read_csv("wpkg_data.csv", index_col='step', nrows=150)
#Get data and as
data = df.values
X, y = data[:, :-1], data[:, -1]
#CyclingElasticNetModel_V2(alpha_set, l1_ratio_set, X, y)

# By running the watts per kg model, it seem model sensitivity is reduced
# To model a real world example, a single exercise session will be used, run at both minute and 1 second itervals

# Session - 1 minute intervals
df = read_csv("wpkg_data_2.csv", index_col='step')
#Get data and as
data = df.values
X, y = data[:, :-1], data[:, -1]
CyclingElasticNetModel_V2(alpha_set, l1_ratio_set, X, y)

## Session - 1 second intervals, limit to 400 seconds max
# df = read_csv("wpkg_data_3.csv", index_col='step', nrows=400)
# #Get data and as
# data = df.values
# X, y = data[:, :-1], data[:, -1]
# CyclingElasticNetModel_V2(alpha_set, l1_ratio_set, X, y)

## 1 minute intervals with added features provides a good basis for future models.

# Demo - Upskilling exercise:
rundemo = False
if rundemo:
    # load dataset
    df = read_csv("housing.csv")

    #Get data and as
    data = df.values
    X, y = data[:, :-1], data[:, -1]
    # define model
    model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = absolute(scores)
    print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
    #--------------------------
    # Tuning
    def ElasticNetCVValues():
        data = df.values
        X, y = data[:, :-1], data[:, -1]
        # define model evaluation method
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # define model
        ratios = arange(0, 1, 0.01)
        alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
        model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
        # fit model
        model.fit(X, y)
        # summarize chosen configuration
        print('alpha: %f' % model.alpha_)
        print('l1_ratio_: %f' % model.l1_ratio_)
        return model.alpha_, model.l1_ratio_
    def ElasticNetValues():
        # Tuning Elastic Net Hyperparameters ElasticNet
        data = df.values
        X, y = data[:, :-1], data[:, -1]
        # define model
        model = ElasticNet()
        # define model evaluation method
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # define grid
        grid = dict()
        grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
        grid['l1_ratio'] = arange(0, 1, 0.01)
        # define search
        search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # perform the search
        results = search.fit(X, y)
        # summarize
        print('MAE: %.3f' % results.best_score_)
        print('Config: %s' % results.best_params_)
        alpha_set = results.best_params_['alpha']
        l1_ratio_set = results.best_params_['l1_ratio']
        return results.best_params_['alpha'], results.best_params_['l1_ratio']

    #-------------------------
    # Use updated params in model

    data = df.values
    X, y = data[:, :-1], data[:, -1]
    # define model
    alpha_set, l1_ratio_set = ElasticNetValues()
    alpha_set_CV, l1_ratio_set_CV  = ElasticNetCVValues()

    def ElasticNetModel(alpha_set, l1_ratio_set, X, y):
        print("running!")
        model = ElasticNet(alpha=alpha_set, l1_ratio=l1_ratio_set)
        # fit model
        model.fit(X, y)
        # define new data
        scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        scores = absolute(scores)
        print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
        values = []
        predictions = []
        for y in range(0,len(data),1):
            for i in range(0, df.shape[1] - 1, 1):
                values.append(df.iloc[y, i])
            # make a prediction
            yhat = model.predict([values])
            predictions.append(yhat)
            # # summarize prediction
            values = []

        # Store Predictions in dataframe
        predictions_df = pd.DataFrame(predictions, columns=["Predicted_MEDV"])
        predictions_att = predictions_df["Predicted_MEDV"].to_numpy()
        MEDV_att = df["MEDV"].to_numpy()

        plt.subplot(111)
        plt.scatter(predictions_att, MEDV_att, linewidth=1.75)
        # adjust
        plt.subplots_adjust(hspace=0.558)
        plt.subplot(111)
        plt.plot([0, max(max(predictions_att),max(MEDV_att))], [0, max(max(predictions_att),max(MEDV_att))], ls="--", c=".3")
        plt.grid(True)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plotshow(True)

    print("Run prior to tuning Elastic Net Hyperparameters ")
    ElasticNetModel(1, 0.5, X, y)
    print("Run after tuning Elastic Net Hyperparameters Using ElasticNet")
    ElasticNetModel(alpha_set, l1_ratio_set, X, y)
    print("Run after tuning Elastic Net Hyperparameters Using ElasticNetCV")
    ElasticNetModel(alpha_set_CV, l1_ratio_set_CV, X, y)
