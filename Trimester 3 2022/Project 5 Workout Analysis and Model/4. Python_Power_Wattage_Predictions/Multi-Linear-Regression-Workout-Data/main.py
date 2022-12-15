import math
import pandas as pd
import numpy as np
## Building Model
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import normal_ad
## Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# show plot and clear
def plotshow(t):
    if t:
        plt.show()
    plt.clf()
    plt.cla()
    plt.close()

### Multi-linear-regression model to predict power
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# Using aquired data  - Predict POWER
# The data set shows average 5 minute values (for a single user) - By expanding the time bracket from
# 1 second to 1 minute to 5 minutes, the data has become less noisey and easier to work with.
# the dataset "train_test" set is use intially to train the model equates to close to 12 hours worth
# of exercise data; or 43,500 seconds. One the 5 minute internal data set tests are completed, a small attempt
# to use a data set where time intervals are based on single seconds will be used to understand the impact
# of expanding the time bracket from 1 second to 300 seconds (5 minutes).

# ---------------
# data sets
df_1 = pd.read_csv("train_test.csv", index_col='step')
df_2 = pd.read_csv("test_set.csv", index_col='step')
df_3 = pd.read_csv("train_test_second.csv", index_col='step')
df_4 = pd.read_csv("test_set_second.csv", index_col='step')
# ---------------
def Multi_linear_regression_model(df_data, df_data_test):
    df_data = df_data  # pd.read_csv("train_test.csv", index_col='step')
    # Data Preprocessing for Classification
    df_data = df_data.reindex(
        columns=["enhanced_speed", "cadence", "max_heart_rate_perct", "power"])
    print(df_data.shape)
    print(df_data.info())

    # Visualize the data using scatter plot and histogram
    sns.set_palette('colorblind')
    sns.pairplot(data=df_data, height=3)
    plotshow(True)

    X = df_data.iloc[:, [1, 2]]
    y = df_data.iloc[:, [3]]

    # Initialize model from sklearn and fit it into our data
    regr = linear_model.LinearRegression()
    model = regr.fit(X.values, y)

    print('Intercept:', model.intercept_)
    print('Coefficients:', model.coef_)

    # y_hat = Intercept - Coefficients1-x_1 + Coefficients2-x_2
    # The intercept value is the estimated average value of our dependent
    # variable when all of our independent variables values is 0

    # predictions bucket
    predictions = []

    # Get a testing set to apply predictions agaisnt
    df_data_test = df_data_test  # pd.read_csv("test_set.csv", index_col='step')
    # Data Preprocessing for Classification
    print("Length of results set: ", len(df_data_test))

    df_data_test = df_data_test.reindex(
        columns=["enhanced_speed", "cadence", "max_heart_rate_perct", "power"])
    # Values to predict
    tune = 1.3
    for i in range(0, len(df_data_test), 1):
        param1 = df_data_test.iat[i, 1]
        param2 = df_data_test.iat[i, 2]
        try:
            predictions.append(int((model.predict([[float(param1), float(param2)]])[0])))
        except ValueError:
            print('Issues, please debug')

    # Prepare data
    X = df_data_test[["cadence", "max_heart_rate_perct"]].values
    Y = df_data_test['power']

    # Create range for each dimension
    x = X[:, 0]
    y = X[:, 1]
    z = Y

    xx_pred = np.linspace(math.floor(min(df_data_test["cadence"])), math.ceil(max(df_data_test["cadence"])),
                          100)  # range of cadence values
    yy_pred = np.linspace(math.floor(min(df_data_test["max_heart_rate_perct"])),
                          math.ceil(max(df_data_test["max_heart_rate_perct"])), 100)  # range of max hr % values
    xx_pred, yy_pred = np.meshgrid(xx_pred, yy_pred)
    model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

    # Predict using model built on previous step
    ols = linear_model.LinearRegression()
    model = ols.fit(X, Y)
    predicted = model.predict(model_viz)

    # Evaluate model by using it's R^2 score
    r2 = model.score(X, Y)
    print("R^2:", r2)

    # Plot model visualisation
    plt.style.use('fivethirtyeight')

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    axes = [ax1, ax2, ax3]

    for ax in axes:
        ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0, 0, 0, 0), s=20, edgecolor='#70b3f0')
        ax.set_xlabel('Cadence', fontsize=12)
        ax.set_ylabel('HR%', fontsize=12)
        ax.set_zlabel('Power', fontsize=12)
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=5, axis='x')

    ax1.view_init(elev=25, azim=-60)
    ax2.view_init(elev=15, azim=15)
    ax3.view_init(elev=25, azim=60)

    fig.suptitle('Multi-Linear Regression Model Visualization ($R^2 = %.2f$)' % r2, fontsize=15, color='k')
    fig.tight_layout()
    plotshow(True)

    # Model Validation
    # We can evaluate a model by looking at it’s coefficient of determination (R²),
    # F-test, t-test, and also residuals. Before we continue we will rebuild our model
    # using the statsmodel library with the OLS() function.
    # Then we will print the model summary using the summary() function on the model.
    # The model summary contains lots of important values we can use to evaluate our model.

    X = df_data_test[["cadence", "max_heart_rate_perct"]]
    X = sm.add_constant(X)  # adding a constant

    olsmod = sm.OLS(df_data_test['power'], X).fit()
    print(olsmod.summary())
    print('R2 score:', olsmod.rsquared)

    # R² range between 0 and 1, where R²=0 means there are no linear relationship between the variables and R²=1 shows
    # a perfect linear relationship. In our case, we got R² score about 0.X which means X% of our dependent variable
    # can be explained using our independent variables.

    # F-Test (ANOVA)
    print('F-statistic:', olsmod.fvalue)
    print('Probability of observing value at least as high as F-statistic:', olsmod.f_pvalue)

    # T-Test
    print(olsmod.pvalues)

    df_data_test['power_prediction'] = predictions
    df_data_test = df_data_test.reindex(
        columns=["enhanced_speed", "cadence", "max_heart_rate_perct", "power", "power_prediction", "residual"])
    df_data_test['power_predict'] = olsmod.predict(X)
    df_data_test['residual'] = olsmod.resid
    print(df_data_test.head())
    df_data_test.to_csv("results.csv")

    # Linearity
    # This assumes that there is a linear relationship between the independent variables
    # and the dependent variable. In our case since we have multiple independent variables,
    # we can do this by using a scatter plot to see our predicted values versus the actual values.
    # Plotting the observed vs predicted values
    sns.lmplot(x='power', y='power_prediction', data=df_data_test, fit_reg=False)

    # Plotting the diagonal line
    line_coords = np.arange(df_data_test[['power', 'power_prediction']].min().min() - 10,
                            df_data_test[['power', 'power_prediction']].max().max() + 10)
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')

    plt.ylabel('Predicted Power', fontsize=14)
    plt.xlabel('Actual Power', fontsize=14)
    plt.title('Linearity Assumption', fontsize=16)
    plotshow(True)

    # Normality
    # This assumes that the error terms of the model are normally distributed.
    # We will examine the normality of the residuals by plotting it into histogram and looking
    # at the p-value from the Anderson-Darling test for normality. We will use the normal_ad()
    # function from statsmodel to calculate our p-value and then compare it to threshold of 0.05,
    # if the p-value we get is higher than the threshold then we can assume that our residual is normally distributed.

    # Performing the test on the residuals
    p_value = normal_ad(df_data_test['residual'])[1]
    print('p-value from the test Anderson-Darling test below 0.05 generally means non-normal:', p_value)

    # Plotting the residuals distribution
    # plt.subplots(figsize=(8, 4))
    # plt.title('Distribution of Residuals', fontsize=18)
    sns.displot(df_data_test['residual'])
    plotshow(True)

    # Reporting the normality of the residuals
    if p_value < 0.05:
        print('Residuals are not normally distributed')
    else:
        print('Residuals are normally distributed')

    # Multicollinearity
    # #This assumes that the predictors used in the regression are not correlated with each other.
    # To identify if there are any correlation between our predictors we can calculate the Pearson correlation coefficient
    # between each column in our data using the corr() function from Pandas dataframe.
    # Then we can display it as a heatmap using heatmap() function from Seaborn.

    corr = df_data_test[['cadence', "max_heart_rate_perct", "power"]].corr()
    print('Pearson correlation coefficient matrix of each variables:\n', corr)

    # Generate a mask for the diagonal cell
    mask = np.zeros_like(corr, dtype=np.bool_)
    np.fill_diagonal(mask, val=True)

    # Initialize matplotlib figure
    fig, ax = plt.subplots(figsize=(4, 3))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)
    cmap.set_bad('grey')

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)
    fig.suptitle('Pearson correlation coefficient matrix', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    fig.tight_layout()
    plotshow(True)

    # The ouput of the heatmap shows us that the independent variables are  ffecting each other and that
    # there is multicollinearity in our data.

    # Autocorrelation
    # Autocorrelation is correlation of the errors (residuals) over time.
    # Used when data are collected over time to detect if autocorrelation is present.
    # Autocorrelation exists if residuals in one time period are related to residuals in another period.
    # We can detect autocorrelation by performing Durbin-Watson test to determine if either positive or
    # negative correlation is present. In this step we will use the durbin_watson () function from statsmodel
    # to calculate our Durbin-Watson score and then assess the value with the following condition:

    # outcomes:
    # If the Durbin-Watson score is less than 1.5 then there is a positive autocorrelation and the assumption is not satisfied
    # If the Durbin-Watson score is between 1.5 and 2.5 then there is no autocorrelation and the assumption is satisfied
    # If the Durbin-Watson score is more than 2.5 then there is a negative autocorrelation and the assumption is not satisfied
    # We can assume that there is Signs of positive autocorrelation in our residual.

    durbinWatson = durbin_watson(df_data_test['residual'])

    print('Durbin-Watson:', durbinWatson)
    if durbinWatson < 1.5:
        print('Signs of positive autocorrelation', '\n')
        print('Assumption not satisfied')
    elif durbinWatson > 2.5:
        print('Signs of negative autocorrelation', '\n')
        print('Assumption not satisfied')
    else:
        print('Little to no autocorrelation', '\n')
        print('Assumption satisfied')

    # Homoscedasticity
    # This assumes homoscedasticity, which is the same variance within our error terms. Heteroscedasticity,
    # the violation of homoscedasticity, occurs when we don’t have an even variance across the error terms.
    # To detect homoscedasticity, we can plot our residual and see if the variance appears to be uniform.

    # Plotting the residuals
    plt.subplots(figsize=(8, 4))
    plt.scatter(x=df_data_test.index, y=df_data_test.residual, alpha=0.8)
    plt.plot(np.repeat(0, len(df_data_test.index) + 2), color='darkorange', linestyle='--')
    plt.ylabel('Residual', fontsize=14)
    plt.title('Homescedasticity Assumption', fontsize=16)
    plotshow(True)
    # Futher analysis is required to validate the homoscedasticity assumption.

    # SUMMARY
    # Our models has failed to passed all the tests in the model validation steps, so we can't conclude that our model can perform well
    # to predict power output using the two independent variables, cadence and max HR %. Our model only has R² score of ~53%, which means
    # that there is still about 47% unknown factors that are affecting our power output. This analysis will inform future efforts
    # to predict power output

    # Visualisation of Predictions (Time series plot)
    plt.subplot(211)
    power_att = df_data_test["power"].to_numpy()
    plt.plot(df_data_test.index.values, power_att, linewidth=1.75)
    # if len(df_data_test) < 2500:
    #     len_x = 1400
    #     len_y = 2400
    # else:
    #     len_x = 120000
    #     len_y = 125000
    # plt.xlim(len_x, len_y)
    plt.subplot(211)
    plt.xlabel("Step")
    plt.ylabel("Watts")
    prediction_att = df_data_test["power_prediction"].to_numpy()
    plt.plot(df_data_test.index.values, prediction_att, linewidth=1.5)
    plt.subplot(212)
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
    plt.subplot(212)
    plt.scatter(power_att, prediction_att)
    plt.title("R^2 = %f" % r2)
    # adjust
    plt.subplots_adjust(hspace=0.558)
    plotshow(True)

# -----------------------
# 5 minute set:
print("5 minute interval set - model running")
Multi_linear_regression_model(df_1, df_1)
# 1 second set:
# print("1 Second interval set - model running")
# Multi_linear_regression_model(df_3, df_4)
# ------------------------
print("\n\nEND OF PROGRAM")
# Resource: https://medium.com/swlh/multi-linear-regression-using-python-44bd0d10082d
