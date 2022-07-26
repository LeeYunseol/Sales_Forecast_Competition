# Importing Necessary Libraries and Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import iplot
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

data = pd.read_csv("train.csv", encoding="utf-8")
data_test = pd.read_csv("test.csv")
print(data.columns)
#%%
# 데이터 정보 확인
data.info()

#%%
# Check negative value of Promotion
print("Train Negative Value Of Pormotion")
print(sum(data['Promotion1'] < 0))
print(sum(data['Promotion2'] < 0))
print(sum(data['Promotion3'] < 0))
print(sum(data['Promotion4'] < 0))
print(sum(data['Promotion5'] < 0))

print("Test Negative Value Of Pormotion")
print(sum(data_test['Promotion1'] < 0))
print(sum(data_test['Promotion2'] < 0))
print(sum(data_test['Promotion3'] < 0))
print(sum(data_test['Promotion4'] < 0))
print(sum(data_test['Promotion5'] < 0))

# There are some negative values on the dataset So I switch these values to 0
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
#%%
# Checking the Histogram of Target Variables
sns.distplot(data['Weekly_Sales'], fit=stats.norm)
# Original Target variables is imbalance, So I used Log transform

sns.distplot(np.log1p(data['Weekly_Sales']), fit=stats.norm)
# It seems balance now

#%%
# Date format is not comportable to me. So i change the format : day/month/year -> year-month-day
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


test_date_df = data_test.loc[:, ['Date']]
for i in range(len(test_date_df)) :
    date_list =  test_date_df.loc[i].str.split('/')
    test_date_df.loc[i] = date_list[0][2] + '-' + date_list[0][1] + '-' + date_list[0][0]

data_test['Date'] = test_date_df

# Change type of 'Date' : object to datedime
data_test['Date'] = pd.to_datetime(data_test["Date"])
data_test['week'] =data_test['Date'].dt.week
data_test['month'] =data_test['Date'].dt.month 
data_test['year'] =data_test['Date'].dt.year
data_test['WeekOfYear'] = (data_test.Date.dt.isocalendar().week)*1.0 
data_test['day'] = data_test['Date'].dt.day


#%%
# 시계열 데이터 군집 분석
from tslearn.clustering import TimeSeriesKMeans

scaled_time_series_df = pd.DataFrame()
for num in range(1, 46) :
    col_name = "Store " +str(num)
    scaler = MinMaxScaler()
    time_series = data[(data.Store==num) & (data.month <= 10)]['Weekly_Sales'].values.reshape(-1, 1)
    scaled_time_series = scaler.fit_transform(time_series) 
    scaled_time_series = pd.DataFrame(scaled_time_series)
    scaled_time_series_df[col_name] = scaled_time_series

transpose_scaled_time_series_df = scaled_time_series_df.transpose()
    
km = TimeSeriesKMeans(n_clusters=2, 
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
        data.loc[(data.Store== i + 1), 'Type'] = 0
        data_test.loc[(data_test.Store == i+1), 'Type'] = 0
    elif prediction[i] == 1 :
        data.loc[(data.Store== i + 1), 'Type'] = 1
        data_test.loc[(data_test.Store == i+1), 'Type'] = 1
    else:
        data.loc[(data.Store== i + 1), 'Type'] = 2
        data_test.loc[(data_test.Store == i+1), 'Type'] = 2

#%%
# Store one-hot encoding
'''
data = pd.get_dummies(data, columns = ['Store'])
data_test = pd.get_dummies(data_test, columns = ['Store'])

data = pd.get_dummies(data, columns = ['Type'])
data_test = pd.get_dummies(data_test, columns = ['Type'])
print("train data shape : {}".format(data.shape))
print("test data shape : {}".format(data_test.shape))
'''
#%%
# Fill 0 on missing value
data.fillna(0, inplace=True)
data_test.fillna(0, inplace = True)

#%%
'''
# Log Transform
# Useless
data['Weekly_Sales'] = np.log1p(data['Weekly_Sales'])


data['Promotion1'] = np.log1p(data['Promotion1'])
data['Promotion2'] = np.log1p(data['Promotion2'])
data['Promotion3'] = np.log1p(data['Promotion3'])
data['Promotion4'] = np.log1p(data['Promotion4'])
data['Promotion5'] = np.log1p(data['Promotion5'])

data_test['Promotion1'] = np.log1p(data_test['Promotion1'])
data_test['Promotion2'] = np.log1p(data_test['Promotion2'])
data_test['Promotion3'] = np.log1p(data_test['Promotion3'])
data_test['Promotion4'] = np.log1p(data_test['Promotion4'])
data_test['Promotion5'] = np.log1p(data_test['Promotion5'])
'''
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

#%%
# 'Temperature', 'Fuel_Price', 'Promotion1',
           #'Promotion2','Promotion3','Promotion4', 'Unemployment', 'month', 'year','day'
data = data.drop(['id', 'Date'], axis = 1)
data_test = data_test.drop(['id', 'Date'], axis = 1)
# Save subset
data.to_csv("train_set.csv", index = False)
data_test.to_csv("test_set.csv", index = False)
#%% 
# 표준화

temp = data["Weekly_Sales"]
scaler = MinMaxScaler()
scaler.fit(data)
data_scaled = scaler.transform(data)
real_data = pd.DataFrame(data=data_scaled, columns= data.columns)

data = data.drop("Weekly_Sales", axis = 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data)
data_scaled = scaler.transform(data)
#data = pd.DataFrame(data=data_scaled, columns= data.columns)

data_test_scaled = scaler.transform(data_test)
data_test = pd.DataFrame(data=data_test_scaled, columns= data.columns)
real_data["Original_Weekly_Sales"] = temp

#%%
# Save subset
real_data.to_csv("minmax_train_set.csv", index = False)
data_test.to_csv("minmax_test_set.csv", index = False)
