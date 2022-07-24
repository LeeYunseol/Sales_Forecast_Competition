import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('train_set.csv')
print(data.columns)
#%%
'''
data_scaler = data.iloc[:, :]
print(data_scaler.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['Store', 'year','IsHoliday', 'Promotion3','WeekOfYear','Weekly_Sales'])

# data_df_scaled['Weekly_Sales'] = data['Weekly_Sales']

data_for_scale = data.iloc[:,:-1]
scaler_for_test = MinMaxScaler()
scaler_for_test.fit(data_for_scale)                                        
data_df_scaled.to_csv("minmax_train_set.csv", index = False)
'''
#%%

# PREPROCESS TEST!!!!!!!!!!!!!!

test = pd.read_csv('test.csv')
print(test.columns)

#test = test.interpolate(method='values')
test = test.fillna(0)

print("test 데이터 결측치 수 : {}".format(test.isna().sum()))

test_df = test.loc[:, ['Date']]


for i in range(len(test_df)) :
    date_list =  test_df.loc[i].str.split('/')
    test_df.loc[i] = date_list[0][2] + '-' + date_list[0][1] + '-' + date_list[0][0]

test['Date'] = test_df

test['Date'] = pd.to_datetime(test["Date"])
test['week'] =test['Date'].dt.week
test['month'] =test['Date'].dt.month 
test['year'] =test['Date'].dt.year
test['day'] = test['Date'].dt.day
test['WeekOfYear'] = (test.Date.dt.isocalendar().week)*1.0 
# Super bowl dates in train set
test.loc[(test['Date'] == '2010-02-12')|(test['Date'] == '2011-02-11')|(test['Date'] == '2012-02-10'),'Super_Bowl'] = True
test.loc[(test['Date'] != '2010-02-12')&(test['Date'] != '2011-02-11')&(test['Date'] != '2012-02-10'),'Super_Bowl'] = False

# Labor day dates in train set
test.loc[(test['Date'] == '2010-09-10')|(test['Date'] == '2011-09-09')|(test['Date'] == '2012-09-07'),'Labor_Day'] = True
test.loc[(test['Date'] != '2010-09-10')&(test['Date'] != '2011-09-09')&(test['Date'] != '2012-09-07'),'Labor_Day'] = False

# Thanksgiving dates in train set
test.loc[(test['Date'] == '2010-11-26')|(test['Date'] == '2011-11-25'),'Thanksgiving'] = True
test.loc[(test['Date'] != '2010-11-26')&(test['Date'] != '2011-11-25'),'Thanksgiving'] = False

#Christmas dates in train set
test.loc[(test['Date'] == '2010-12-31')|(test['Date'] == '2011-12-30'),'Christmas'] = True
test.loc[(test['Date'] != '2010-12-31')&(test['Date'] != '2011-12-30'),'Christmas'] = False

test['Thanksgiving'] = test['Thanksgiving'].astype(bool).astype(int) # changing T,F to 0-1
test['Christmas'] = test['Christmas'].astype(bool).astype(int) # changing T,F to 0-1
test['IsHoliday'] = test['IsHoliday'].astype(bool).astype(int) # changing T,F to 0-1
test['Super_Bowl'] = test['Super_Bowl'].astype(bool).astype(int) # changing T,F to 0-1
test['Labor_Day'] = test['Labor_Day'].astype(bool).astype(int) # changing T,F to 0-1
#%%
# 잠시만 앞에서 했던 것
train = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/train_data/minmax_train_set.csv')
from tslearn.clustering import TimeSeriesKMeans

scaled_time_series_df = pd.DataFrame()
for num in range(1, 46) :
    col_name = "Store " +str(num)
    scaler = MinMaxScaler()
    time_series = train[(train.Store==num) & (train.month <= 10)]['Weekly_Sales'].values.reshape(-1, 1)
    scaled_time_series = scaler.fit_transform(time_series) 
    scaled_time_series = pd.DataFrame(scaled_time_series)
    scaled_time_series_df[col_name] = scaled_time_series

transpose_scaled_time_series_df = scaled_time_series_df.transpose()
    
km = TimeSeriesKMeans(n_clusters=3, 
                      metric="dtw", 
                      max_iter=5,
                      random_state=2022)

prediction = km.fit_predict(transpose_scaled_time_series_df)

for i in range(len(prediction)) :
    if prediction[i] == 0 :
        test.loc[(test.Store== i + 1), 'Type'] = 0
    elif prediction[i] == 1 :
        test.loc[(test.Store== i + 1), 'Type'] = 1
    else:
        test.loc[(test.Store== i + 1), 'Type'] = 2
#%%
#test = test.drop(['month', 'week'], axis=1)
test.to_csv("minmax_test_set.csv", index = False)
# test = test[['Store', 'year','IsHoliday', 'Promotion3', 'WeekOfYear']]

# no min max 
# 


#tst_scaled = scaler_for_test.transform(test)
#data_df_scaled2 = pd.DataFrame(data=test_scaled, columns=['Store', 'year','Thanksgiving', 'Promotion3','WeekOfYear'])

                                         
#data_df_scaled2.to_csv("minmax_test_set.csv", index = False)

