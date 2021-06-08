import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt

base = pd.read_csv(r"C:\Users\spalves\Desktop\Silvia\Pessoal\ITAU\ALL+CSV+FILES\ALL CSV FILES\Hitters.csv")
print("Number of null values:", base["Salary"].isnull().sum())

# Print the dimensions of the original Hitters data (322 rows x 20 columns)
print("Dimensions of original data:", base.shape)

# Drop any rows the contain missing values, along with the player names
base_not_null = base.dropna()

# Print the dimensions of the modified Hitters data (263 rows x 20 columns)
print("Dimensions of modified data:", base_not_null.shape)

# One last check: should return 0
print("Number of null values:", base_not_null["Salary"].isnull().sum())

dummies = pd.get_dummies(base_not_null[['League', 'Division', 'NewLeague']])

y = base_not_null.Salary

# Drop the column with the independent variable (Salary), and columns for which we created dummy variables
X_ = base_not_null.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

# Define the feature set X.
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)


print("\n--------Best Subset Selection by Adjusted R^2, AIC, BIC--------\n")


def processSubset(feature_set):
    
    X_ = sm.add_constant(X[list(feature_set)])
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(y,X_)
    regr = model.fit()
    RSS = ((regr.predict(X_) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}

def getBest(k):
    
    tic = time.time()
    
    results = []
    
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(combo))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the minimum RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    toc = time.time()
    print("Processed", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model

# Could take quite awhile to complete...

models_best = pd.DataFrame(columns=["RSS", "model"])

tic = time.time()
for i in range(1,8):
    models_best.loc[i] = getBest(i)

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")
#%%
print(models_best.loc[6, "model"].summary())

# print(getBest(19)["model"].summary())

models_best.loc[2, "model"].rsquared

# Gets the second element from each row ('model') and pulls out its rsquared attribute
models_best.apply(lambda row: row[1].rsquared, axis=1)

plt.figure(figsize=(10,5))
plt.rcParams.update({'font.size': 10, 'lines.markersize': 10})

# Set up a 2x2 grid so we can look at 4 plots at once
# First plot: RSS

plt.subplot(2, 2, 1)

plt.plot(models_best["RSS"])
plt.xlabel('# Predictors')
plt.ylabel('RSS')

# Second plot: R^2 adjusted
# We will now plot a red dot to indicate the model with the largest adjusted R^2 statistic.
# The argmax() function can be used to identify the location of the maximum point of a vector

rsquared_adj = models_best.apply(lambda row: row[1].rsquared_adj, axis=1)

plt.subplot(2, 2, 2)
plt.plot(rsquared_adj)
plt.plot(rsquared_adj.argmax(), rsquared_adj.max(), "or")
plt.xlabel('# Predictors')
plt.ylabel('adjusted rsquared')

# We'll do the same for AIC and BIC, this time looking for the models with the SMALLEST statistic
aic = models_best.apply(lambda row: row[1].aic, axis=1)

plt.subplot(2, 2, 3)
plt.plot(aic)
plt.plot(aic.argmin(), aic.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('AIC')

bic = models_best.apply(lambda row: row[1].bic, axis=1)

plt.subplot(2, 2, 4)
plt.plot(bic)
plt.plot(bic.argmin(), bic.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('BIC')

#%%
print("\n--------Forward Stepwise Selection--------\n")


def forward(predictors):

    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X.columns if p not in predictors]
    
    tic = time.time()
    
    results = []
    
    for p in remaining_predictors:
        results.append(processSubset(predictors+[p]))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model

models_fwd = pd.DataFrame(columns=["RSS", "model"])

tic = time.time()
predictors = []

for i in range(1,len(X.columns)+1):    
    models_fwd.loc[i] = forward(predictors)
    predictors = models_fwd.loc[i]["model"].model.exog_names[1:]

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")

print(models_fwd.loc[1, "model"].summary())
print(models_fwd.loc[2, "model"].summary())

print(models_best.loc[6, "model"].summary())
print(models_fwd.loc[6, "model"].summary())

#%%
print("\n--------Backward Stepwise Selection--------\n")

def backward(predictors):
    
    tic = time.time()
    
    results = []
    
    for combo in itertools.combinations(predictors, len(predictors)-1):
        results.append(processSubset(combo))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)-1, "predictors in", (toc-tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model

models_bwd = pd.DataFrame(columns=["RSS", "model"], index = range(1,len(X.columns)))

tic = time.time()
predictors = X.columns

while(len(predictors) > 1):  
    models_bwd.loc[len(predictors)-1] = backward(predictors)
    predictors = models_bwd.loc[len(predictors)-1]["model"].model.exog_names[1:]

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")

print("------------")
print("Best Subset:")
print("------------")
print(models_best.loc[6, "model"].params)

print("-----------------")
print("Foward Selection:")
print("-----------------")
print(models_fwd.loc[6, "model"].params)

print("-------------------")
print("Backward Selection:")
print("-------------------")
print(models_bwd.loc[6, "model"].params)

print("------------")
print("Best Subset:")
print("------------")
print(models_best.loc[7, "model"].params)

print("-----------------")
print("Foward Selection:")
print("-----------------")
print(models_fwd.loc[7, "model"].params)

print("-------------------")
print("Backward Selection:")
print("-------------------")
print(models_bwd.loc[7, "model"].params)