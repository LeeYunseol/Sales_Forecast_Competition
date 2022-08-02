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
from sklearn import metrics, ensemble, linear_model
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/minmax_train_set.csv')
test_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/minmax_test_set.csv')

original_train_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/train_set.csv')
original_test_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/test_set.csv')

scaler_for_weekly_sales = MinMaxScaler()
scaler_for_weekly_sales.fit(train_df['Original_Weekly_Sales'].values.reshape(-1, 1))

# X_train
train_2010 = train_df[(train_df.year==0) & (train_df.month<= 0.8)]
train_2011 = train_df[(train_df.year==0.5) & (train_df.month<=0.8)]
train_2012 = train_df[(train_df.year==1) & (train_df.month<=0.7)]

X_train = pd.concat([train_2010, train_2011, train_2012])
y_train = X_train.Weekly_Sales

X_train = X_train.drop(["Weekly_Sales", "Original_Weekly_Sales"], axis = 1)

X_test = train_df[(train_df.year==1) & (train_df.month >= 0.7) & (train_df.month <= 0.8)]
y_test = X_test.Weekly_Sales
X_test_temp = X_test
X_test = X_test.drop(["Weekly_Sales", "Original_Weekly_Sales"], axis = 1)


# 꼭 있어야하는 정보 
#X_train = train_df
#y_train = train_df.Weekly_Sales
feature = list(train_df.columns)
feature.remove('Weekly_Sales')
feature.remove('Original_Weekly_Sales')
feature.remove('month')
feature.remove('Type')
#feature = ['Store', 'Type', 'year', 'WeekOfYear','Thanksgiving', 'month', 'day']
#features = ['Store', 'Type', 'IsHoliday', 'year', 'WeekOfYear', 'month', 'day', 'Promotion1',
#             'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Fuel_Price']

#train = train_df

# 그리드 서치를 통해서 성능을 높여보자!!
parameters = {
              'objective':['reg:squarederror'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [10,15],
              'min_child_weight': [4],
              'subsample': [0.8],
              'colsample_bytree': [0.8],
              'n_estimators':[400,500,600,1000]
              } #540??

fit_params={
            "early_stopping_rounds" :50,
            "eval_metric" : "rmse", 
            "eval_set" : [[X_test[feature], y_test]]
            }


xgb = XGBRegressor(random_state = 2022)
xgb.set_params(**fit_params)
xgb_grid = GridSearchCV(xgb,
                        parameters,
                        cv = 10,
                        scoring = 'neg_mean_absolute_error',
                        n_jobs = 5,
                        verbose=3
                        )
#  **fit_params
xgb_grid.fit(X_train[feature], y_train, eval_set = [[X_test[feature], y_test]])#, **fit_params)
best_model = xgb_grid.best_estimator_


print("BEST SCORE : {}".format(xgb_grid.best_score_))
print("BEST PARAMETER : {}".format(xgb_grid.best_params_))


'''
model = XGBRegressor(colsample_bytree=0.8, max_depth= 5, learning_rate= 0.25, n_estimators=500,
                   random_state =2022, nthread = -1, n_jobs=-1)

model.fit(train[features], train.Weekly_Sales,
          eval_set=[(train[features], train.Weekly_Sales)],
          eval_metric='rmse', 
          early_stopping_rounds=50,
          verbose = 3)

results = best_model.evals_result()

epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='TEST')
#ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show() 
'''
#%%

pred = best_model.predict(X_test[feature])
x = np.sqrt(metrics.mean_squared_error(y_test, pred))
print("전체")
print("RMSE: ", x) #RMSE
score = r2_score(y_test, pred)
print("R^2 : {}\n".format(score))


# 각 Store의 점수 보기 
for i in range(45) :
    print("Store {}".format(i+1))
    y_test = X_test_temp.iloc[i*4 : i*4+4, :].Weekly_Sales
    temp = X_test_temp.iloc[i*4:i*4+4, :][feature]
    y_pred = best_model.predict(temp)
    x = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print("RMSE : ", x) #RMSE
    score = r2_score(y_test, y_pred)
    print("R^2 : {}\n".format(score))

#%%
# 테스트 후 전체 셋에 대해서 학습
X_train = train_df[train_df.month <= 0.85]
y_train = X_train.Weekly_Sales

#eval_set = [(X_train[feature], X_train.Weekly_Sales)]
#best_model =  xgb_grid.best_estimator_ 
real_best_model = XGBRegressor(random_state = 2022)
real_best_model.set_params(**xgb_grid.best_params_)
new_params = {"n_estimators":800}
real_best_model.set_params(**new_params)
print(real_best_model)  
real_best_model.fit(X_train[feature], X_train.Weekly_Sales)#,
               #eval_set=eval_set,
               #eval_metric='rmse', 
               #early_stopping_rounds=20,
               #verbose = 3
               #)
 #%%
# 학습 종료 후 test.csv 평가
#pred_before = xgb_grid.best_estimator_.predict(test_df[feature])
#pred = np.expm1(xgb_grid.best_estimator_.predict(test_df[feature]))
# pred = model.predict(test_df[features])
prediction = real_best_model.predict(test_df[feature])
prediction = scaler_for_weekly_sales.inverse_transform(prediction.reshape(-1, 1))
prediction = np.expm1(prediction)
test_df["Weekly_Sales"] = prediction
original_test_df["Weekly_Sales"] = prediction

#%%

fig = plt.figure(figsize=(30,60))
plt.title(feature)
for store in range(1,46):
    storeset = original_train_df[original_train_df.Store==store]
    storeset_2010 = storeset[(storeset.year==2010) & (storeset.month<=10)]
    storeset_2011 = storeset[(storeset.year==2011) & (storeset.month<=10)]
    storeset_2012 = storeset[(storeset.year==2012) & (storeset.month<=9)]
    
    test_pred_store = original_test_df[original_test_df.Store==store]
    
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
feature_importance = real_best_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(40, 20))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_train[feature].columns)[sorted_idx])
plt.title('Feature Importance')
#%%
sample_submission = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/sample_submission.csv')
sample_submission["Weekly_Sales"] = test_df.Weekly_Sales
sample_submission.to_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/submission.csv',index = False)

    