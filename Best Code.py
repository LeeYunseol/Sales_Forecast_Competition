#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import iplot
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("train.csv", encoding="utf-8")
data_test = pd.read_csv('test.csv')

print("Train subset column : {}".format(data.columns))
print("Test subset column : {}".format(data_test.columns))


# # 모든 Store 판매 그래프 보기



# In[3]:


print("Train set의 Promotion의 음수값")
print(sum(data['Promotion1'] < 0))
print(sum(data['Promotion2'] < 0))
print(sum(data['Promotion3'] < 0))
print(sum(data['Promotion4'] < 0))
print(sum(data['Promotion5'] < 0))

print("Test set의 Promotion의 음수값")
print(sum(data_test['Promotion1'] < 0))
print(sum(data_test['Promotion2'] < 0))
print(sum(data_test['Promotion3'] < 0))
print(sum(data_test['Promotion4'] < 0))
print(sum(data_test['Promotion5'] < 0))


# 음수 값이 있는 것을 확인하였고 Promotion 값은 음수가 있으면 안된다고 판단하였기에 이를 0으로 변환

# # 결측치 확인




data['Promotion1'][data['Promotion1'] < 0] = 0
data['Promotion2'][data['Promotion2'] < 0] = 0
data['Promotion3'][data['Promotion3'] < 0] = 0
data['Promotion4'][data['Promotion4'] < 0] = 0
data['Promotion5'][data['Promotion5'] < 0] = 0

data_test['Promotion1'][data_test['Promotion1'] < 0] = 0
data_test['Promotion2'][data_test['Promotion2'] < 0] = 0
data_test['Promotion3'][data_test['Promotion3'] < 0] = 0
data_test['Promotion4'][data_test['Promotion4'] < 0] = 0
data_test['Promotion5'][data_test['Promotion5'] < 0] = 0

# Fill 0 on missing value
data.fillna(0, inplace=True)
data_test.fillna(0, inplace = True)


# # Date Format 변환

# In[6]:


# 일/월/년도 식의 format은 보기 분편해서 임의로 편하게 바꿔주었다.
date_df = data.loc[:, ['Date']]


for i in range(len(date_df)) :
    date_list =  date_df.loc[i].str.split('/')
    date_df.loc[i] = date_list[0][2] + '-' + date_list[0][1] + '-' + date_list[0][0]

data['Date'] = date_df

# Change type of 'Date' : object to datedime
data['Date'] = pd.to_datetime(data["Date"])
data['week'] =data['Date'].dt.week
data['month'] =data['Date'].dt.month 
data['year'] =data['Date'].dt.year
data['WeekOfYear'] = (data.Date.dt.isocalendar().week)*1.0 
data['day'] = data['Date'].dt.day


date_df = data_test.loc[:, ['Date']]


for i in range(len(date_df)) :
    date_list =  date_df.loc[i].str.split('/')
    date_df.loc[i] = date_list[0][2] + '-' + date_list[0][1] + '-' + date_list[0][0]

data['Date'] = date_df
# Change type of 'Date' : object to datedime
data_test['Date'] = pd.to_datetime(data_test["Date"])
data_test['week'] =data_test['Date'].dt.week
data_test['month'] =data_test['Date'].dt.month 
data_test['year'] =data_test['Date'].dt.year
data_test['WeekOfYear'] = (data_test.Date.dt.isocalendar().week)*1.0 
data_test['day'] = data_test['Date'].dt.day


# # 각 Store의 연도별 판매량 확인

# In[7]:





from tslearn.clustering import TimeSeriesKMeans

scaled_time_series_df = pd.DataFrame()
for num in range(1, 46) :
    col_name = "Store " +str(num)
    scaler = MinMaxScaler()
    # test 셋은 2012년 10월 데이터이기 때문에 각 연도별 10월(포함) 이전의 시계열 데이터의 유사성을 판단하여 type을 나누었습니다.
    time_series = data[(data.Store==num) & (data.month <= 10)]['Weekly_Sales'].values.reshape(-1, 1)
    scaled_time_series = scaler.fit_transform(time_series) 
    scaled_time_series = pd.DataFrame(scaled_time_series)
    scaled_time_series_df[col_name] = scaled_time_series

transpose_scaled_time_series_df = scaled_time_series_df.transpose()
    
km = TimeSeriesKMeans(n_clusters=3, 
                      metric="dtw", 
                      max_iter=5,
                      random_state=2022)

prediction = km.fit_predict(transpose_scaled_time_series_df)

list_0 = []
list_1 = []
list_2 = []

for i in range(len(prediction)) :
    if prediction[i] == 0 :
        list_0.append(i+1)
    elif prediction[i] == 1 :
        list_1.append(i+1)
    else:
        list_2.append(i+1)

print("Clustering 0 : ", list_0)
print("Clustering 1 : ", list_1)
print("Clustering 2 : ", list_2)

for i in range(len(prediction)) :
    if prediction[i] == 0 :
        data.loc[(data.Store == i + 1), 'Type'] = 0
        data_test.loc[(data_test.Store == i + 1), 'Type'] = 0
    elif prediction[i] == 1 :
        data.loc[(data.Store == i + 1), 'Type'] = 1
        data_test.loc[(data_test.Store == i + 1), 'Type'] = 1
    else:
        data.loc[(data.Store == i + 1), 'Type'] = 2
        data_test.loc[(data_test.Store == i + 1), 'Type'] = 2


# 3개로 군집화 했을 때가 2개, 4개, 5개로 했을 때보다 더 좋아서 3개로 하였습니다.

# # 공휴일 구분하기

# In[11]:


#%%
data.loc[(data['Date'] == '2010-02-12')|(data['Date'] == '2011-02-11')|(data['Date'] == '2012-02-10'),'Super_Bowl'] = True
data.loc[(data['Date'] != '2010-02-12')&(data['Date'] != '2011-02-11')&(data['Date'] != '2012-02-10'),'Super_Bowl'] = False

# Labor day dates in train set
data.loc[(data['Date'] == '2010-09-10')|(data['Date'] == '2011-09-09')|(data['Date'] == '2012-09-07'),'Labor_Day'] = True
data.loc[(data['Date'] != '2010-09-10')&(data['Date'] != '2011-09-09')&(data['Date'] != '2012-09-07'),'Labor_Day'] = False

# Thanksgiving dates in train set
data.loc[(data['Date'] == '2010-11-26')|(data['Date'] == '2011-11-25'),'Thanksgiving'] = True
data.loc[(data['Date'] != '2010-11-26')&(data['Date'] != '2011-11-25'),'Thanksgiving'] = False

#Christmas dates in train set
data.loc[(data['Date'] == '2010-12-31')|(data['Date'] == '2011-12-30'),'Christmas'] = True
data.loc[(data['Date'] != '2010-12-31')&(data['Date'] != '2011-12-30'),'Christmas'] = False

data['Super_Bowl'] = data['Super_Bowl'].astype(bool).astype(int) # changing T,F to 0-1
data['Thanksgiving'] = data['Thanksgiving'].astype(bool).astype(int) # changing T,F to 0-1
data['Labor_Day'] = data['Labor_Day'].astype(bool).astype(int) # changing T,F to 0-1
data['Christmas'] = data['Christmas'].astype(bool).astype(int) # changing T,F to 0-1
data['IsHoliday'] = data['IsHoliday'].astype(bool).astype(int) # changing T,F to 0-1
#%%

data_test.loc[(data_test['Date'] == '2010-02-12')|(data_test['Date'] == '2011-02-11')|(data_test['Date'] == '2012-02-10'),'Super_Bowl'] = True
data_test.loc[(data_test['Date'] != '2010-02-12')&(data_test['Date'] != '2011-02-11')&(data_test['Date'] != '2012-02-10'),'Super_Bowl'] = False

# Labor day dates in train set
data_test.loc[(data_test['Date'] == '2010-09-10')|(data_test['Date'] == '2011-09-09')|(data_test['Date'] == '2012-09-07'),'Labor_Day'] = True
data_test.loc[(data_test['Date'] != '2010-09-10')&(data_test['Date'] != '2011-09-09')&(data_test['Date'] != '2012-09-07'),'Labor_Day'] = False

# Thanksgiving dates in train set
data_test.loc[(data_test['Date'] == '2010-11-26')|(data_test['Date'] == '2011-11-25'),'Thanksgiving'] = True
data_test.loc[(data_test['Date'] != '2010-11-26')&(data_test['Date'] != '2011-11-25'),'Thanksgiving'] = False

#Christmas dates in train set
data_test.loc[(data_test['Date'] == '2010-12-31')|(data_test['Date'] == '2011-12-30'),'Christmas'] = True
data_test.loc[(data_test['Date'] != '2010-12-31')&(data_test['Date'] != '2011-12-30'),'Christmas'] = False

data_test['Super_Bowl'] = data_test['Super_Bowl'].astype(bool).astype(int) # changing T,F to 0-1
data_test['Thanksgiving'] = data_test['Thanksgiving'].astype(bool).astype(int) # changing T,F to 0-1
data_test['Labor_Day'] = data_test['Labor_Day'].astype(bool).astype(int) # changing T,F to 0-1
data_test['Christmas'] = data_test['Christmas'].astype(bool).astype(int) # changing T,F to 0-1
data_test['IsHoliday'] = data_test['IsHoliday'].astype(bool).astype(int) # changing T,F to 0-1



data.to_csv("train_set.csv", index = False)
data_test.to_csv("test_set.csv", index = False)


# In[34]:


data["Weekly_Sales"] = np.log1p(data["Weekly_Sales"]) 


# # 학습을 위한 전처리

# ### MinMaxScaler

# In[35]:


data = data.drop([ 'id', 'Date', 'WeekOfYear'], axis = 1)
data_test = data_test.drop(['id', 'Date', 'WeekOfYear'], axis = 1)

temp = data["Weekly_Sales"]
scaler = MinMaxScaler()

scaler.fit(data)
data_scaled = scaler.transform(data)
real_data = pd.DataFrame(data=data_scaled, columns= data.columns)

data = data.drop("Weekly_Sales", axis = 1)
from sklearn.preprocessing import MinMaxScaler
scaler2 = MinMaxScaler()
scaler2.fit(data)
data_scaled = scaler2.transform(data)

data_test_scaled = scaler2.transform(data_test)
data_test = pd.DataFrame(data=data_test_scaled, columns= data_test.columns)
real_data["Original_Weekly_Sales"] = temp

# Save subset
real_data.to_csv("minmax_train_set.csv", index = False)
data_test.to_csv("minmax_test_set.csv", index = False)


# # 학습

# In[36]:


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

train_df = pd.read_csv('minmax_train_set.csv')
test_df = pd.read_csv('minmax_test_set.csv')

# 스케일러를 위한 Original train set 불러오기
original_train_df = pd.read_csv('train_set.csv')
original_test_df = pd.read_csv('test_set.csv')

scaler_for_weekly_sales = MinMaxScaler()
scaler_for_weekly_sales.fit(train_df['Original_Weekly_Sales'].values.reshape(-1, 1))

# X_train
X_train = train_df[train_df.month <= 0.85]
y_train = X_train.Weekly_Sales

# 앞의분 분석에서 중요하다고 생각했던 feature들을 따로 모았습니다. 
feature = ['Store', 'Type', 'year', 'week','IsHoliday', 'month', 'day']


# 날짜와 관련된 feature를 사용하는게 예측 성능이 가장 좋았습니다. <br>
# 특히 Promotion에 해당하는 feature들을 사용하면 예측 성능이 오히려 떨어졌습니다. 그 이유는 Promotion이 진행됐던 해는 2011년 4분기이기 때문으로 추측됩니다.

# ### GridSearch

# In[37]:


parameters = {
              'objective':['reg:squarederror'],
              'learning_rate':[0.1], #so called `eta` value
              'max_depth': [50],
              'min_child_weight': [4],
              'subsample': [0.8],
              'colsample_bytree': [0.8],
              'n_estimators':[30000]
              } 

xgb = XGBRegressor(random_state = 2022)

xgb_grid = GridSearchCV(xgb,
                        parameters,
                        cv = 10,
                        scoring = 'neg_mean_absolute_error',
                        n_jobs = 5,
                        verbose=3
                        )

xgb_grid.fit(X_train[feature], y_train)
best_model = xgb_grid.best_estimator_
print("BEST SCORE : {}".format(xgb_grid.best_score_))
print("BEST PARAMETER : {}".format(xgb_grid.best_params_))


# 토크에서 이런 예측 대회로는 과적합이 가장 좋다고해서 육안으로 max_depth와 n_estimator가 최대가 되는 파라미터를 찾았습니다. 그 값들이 이 값들이었습니다. 많은 시행 착오가 있었지만 가장 좋았던 파라미터만을 이 코드에서는 제시하겠습니다.

# ### Best Model의 예측값을 저장

# In[38]:


prediction = best_model.predict(test_df[feature])
prediction = scaler_for_weekly_sales.inverse_transform(prediction.reshape(-1, 1))
prediction = np.expm1(prediction)
test_df["Weekly_Sales"] = prediction
original_test_df["Weekly_Sales"] = prediction


# In[42]:


print(prediction)


# ### 그래프를 그려 예측 성능 육안으로 확인

# In[39]:


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


# # Feature importance 확인

# In[40]:


feature_importance = best_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(40, 20))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_train[feature].columns)[sorted_idx])
plt.title('Feature Importance')


# # submission 저장

# In[41]:


sample_submission = pd.read_csv('sample_submission.csv')
sample_submission["Weekly_Sales"] = test_df.Weekly_Sales
sample_submission.to_csv('submission.csv',index = False)

