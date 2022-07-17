import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
train_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/train_data/minmax_train_set.csv')
test_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/prepossed_test_data/minmax_test_set.csv')

train_X = train_df.drop(["Weekly_Sales"], axis=1)
train_y = train_df["Weekly_Sales"]


X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2, shuffle=False, stratify=None)


etr_random_best = ExtraTreesRegressor(bootstrap=False, criterion="mse", max_depth=None,
                                      max_features="auto", max_leaf_nodes=None,
                                      min_impurity_decrease=0.0,
                                      min_samples_leaf=2, min_samples_split=5,
                                      min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=30,
                                      oob_score=False, random_state=None, warm_start=False)
etr_random_best.fit(X_train, y_train)

y_pred = etr_random_best.predict(X_test)

# Print out the MAE, MSE & RMSE
print("MAE: ", metrics.mean_absolute_error(y_test, y_pred)) #MAE
print("MSE: ", metrics.mean_squared_error(y_test, y_pred)) #MSE
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred))) #RMSE

# rSquared
score = r2_score(y_test, y_pred)
print("R^2:", score)

predict = etr_random_best.predict(test_df)
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['Weekly_Sales'] = predict
sample_submission.to_csv('submission.csv',index = False)
#%%
model = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=400, max_depth=15, learning_rate=0.35)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print out the MAE, MSE & RMSE
print("MAE: ", metrics.mean_absolute_error(y_test, y_pred)) #MAE
print("MSE: ", metrics.mean_squared_error(y_test, y_pred)) #MSE
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred))) #RMSE
predict = model.predict(test_df)
predict = scaler.inverse_transform(predict.reshape(-1, 1))
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['Weekly_Sales'] = predict
sample_submission.to_csv('submission.csv',index = False)

#%%
xgb1 = XGBRegressor()
parameters = { #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [0.3, 0.5, 0.7, 0.1], #so called `eta` value
              'max_depth': [10, 15, 20, 25],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500, 1000, 1500, 2000]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 5,
                        scoring ='neg_mean_absolute_error',
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(X_train, y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
#%%
model=xgb_grid.best_estimator_  
y_pred = model.predict(X_test)

# Print out the MAE, MSE & RMSE
print("MAE: ", metrics.mean_absolute_error(y_test, y_pred)) #MAE
print("MSE: ", metrics.mean_squared_error(y_test, y_pred)) #MSE
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred))) #RMSE
predict = model.predict(test_df)
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['Weekly_Sales'] = predict
sample_submission.to_csv('submission.csv',index = False)