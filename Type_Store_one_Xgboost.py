import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
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
from sklearn import metrics, ensemble, linear_model
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/train_data/minmax_train_set.csv')
test_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/prepossed_test_data/minmax_test_set.csv')


practice_features = ['Store', 'Type', 'IsHoliday', 'year', 'WeekOfYear', 'day', 'month']
#practice_features = ['Store', 'Type', 'IsHoliday', 'year', 'WeekOfYear', 'month', 'day', 'Promotion1',
#             'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5']
temp1 = train_df[(train_df.year == 2010) & (train_df.month <= 9)]
temp2 = train_df[(train_df.year == 2011) & (train_df.month <= 9)]
temp3 = train_df[(train_df.year == 2012) & (train_df.month <= 8)]
practice_train = pd.concat([temp1, temp2, temp3])

practice_test = train_df[(train_df.year==2012) & (train_df.month == 9)]

model = XGBRegressor(colsample_bytree=0.8, max_depth= 5, learning_rate= 0.25, n_estimators=500,
                   random_state =2022, nthread = -1, n_jobs=-1)

model.fit(practice_train[practice_features], practice_train.Weekly_Sales,
          eval_set=[(practice_train[practice_features], practice_train.Weekly_Sales), (practice_test[practice_features], practice_test.Weekly_Sales)],
          #eval_set = [(practice_train[practice_features], practice_train.Weekly_Sales)],
          eval_metric=['rmse'], 
          early_stopping_rounds=50,
          verbose = 3)
prac_y_pred = model.predict(practice_test[practice_features])
RMSE = np.sqrt(metrics.mean_squared_error(practice_test.Weekly_Sales, prac_y_pred))
score = r2_score(practice_test.Weekly_Sales, prac_y_pred)
print("R2 : {}".format(score))
print("RMSE : {}".format(RMSE))

# evaluate predictions
# retrieve performance metrics
results = model.evals_result()

epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()

'''
for store in range(1, 46) :
    store_test = train_df[(train_df.Store == store) & (train_df.year == 2012) & (train_df.month == 9)]
    pred = model.predict(store_test[practice_features])
    rmse = np.sqrt(metrics.mean_squared_error(pred, store_test.Weekly_Sales))
    r2 = r2_score(pred, store_test.Weekly_Sales)
    print("Store {}".format(store))
    print("RMSE : {}    R2 : {}\n".format(rmse, r2))
'''
#%%

# 꼭 있어야하는 정보 
features = ['Store', 'Type', 'IsHoliday', 'year', 'WeekOfYear', 'month', 'day']


train = train_df[(train_df.month<=10)]

model = XGBRegressor(colsample_bytree=0.8, max_depth= 5, learning_rate= 0.25, n_estimators=500,
                   random_state =2022, nthread = -1, n_jobs=-1)

model.fit(train[features], train.Weekly_Sales,
          eval_set=[(train[features], train.Weekly_Sales)],
          eval_metric='rmse', 
          early_stopping_rounds=50,
          verbose = 3)
results = model.evals_result()

epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
#ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()               
#%%
# 학습 종료 후 test.csv 평가

pred = model.predict(test_df[features])

test_df["Weekly_Sales"] = pred
#%%
fig = plt.figure(figsize=(30,60))

for store in range(1,46):
    storeset = train_df[train_df.Store==store]
    storeset_2010 = storeset[(storeset.year==2010) & (storeset.WeekOfYear<=43)]
    storeset_2011 = storeset[(storeset.year==2011) & (storeset.WeekOfYear<=43)]
    storeset_2012 = storeset[(storeset.year==2012) & (storeset.WeekOfYear<=43)]
    
    test_pred_store = test_df[test_df.Store==store]
    
    # 그래프의 연속성을 위해 예측한 데이터의 전 주의 데이터도 넣어준다.
    test_pred_store = pd.concat([storeset_2012.iloc[-1:], test_pred_store])
    
    ax = fig.add_subplot(12, 4, store)
    
    plt.title(f"store_{store}")
    ax.plot(storeset_2010.WeekOfYear, storeset_2010.Weekly_Sales, label="2010", alpha=0.3)
    ax.plot(storeset_2011.WeekOfYear, storeset_2011.Weekly_Sales, label="2011", alpha=0.3)
    ax.plot(storeset_2012.WeekOfYear, storeset_2012.Weekly_Sales, label="2012", color='r')
    ax.plot(test_pred_store.WeekOfYear, test_pred_store.Weekly_Sales, label="2012-pred", color='b')
    ax.legend()
    
plt.show()
#%%
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission["Weekly_Sales"] = test_df.Weekly_Sales
sample_submission.to_csv('submission.csv',index = False)
sample_submission

    