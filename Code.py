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

data = pd.read_csv("train.csv", encoding="utf-8")
print(data.columns)

#%%
from matplotlib import dates

fig = plt.figure(figsize=(50,50)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정

for i in range(1,46):
    train2 = data[data.Store == i]

    train2  = train2[["Date", "Weekly_Sales"]]
    
    ax = fig.add_subplot(10,10,i) ## 그림 뼈대(프레임) 생성


    plt.title("store_{}".format(i)) 
    plt.ylabel('Weekly_Sales')
    plt.xticks(rotation=15)
    ax.xaxis.set_major_locator(dates.MonthLocator(interval = 2))
    ax.plot(train2["Date"], train2["Weekly_Sales"],marker='',label='train', color="blue")

plt.show()
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
#%%
fig = plt.figure(figsize=(30,60))
train_df = data
for store in range(1,46):
    storeset = train_df[train_df.Store==store]
    storeset_2010 = storeset[(storeset.year==2010) & (storeset.WeekOfYear<=43)]
    storeset_2011 = storeset[(storeset.year==2011) & (storeset.WeekOfYear<=43)]
    storeset_2012 = storeset[(storeset.year==2012) & (storeset.WeekOfYear<=43)]
    
    #test_pred_store = test_df[test_df.Store==store]
    
    # 그래프의 연속성을 위해 예측한 데이터의 전 주의 데이터도 넣어준다.
    #test_pred_store = pd.concat([storeset_2012.iloc[-1:], test_pred_store])
    
    ax = fig.add_subplot(12, 4, store)
    
    plt.title(f"store_{store}")
    ax.plot(storeset_2010.WeekOfYear, storeset_2010.Weekly_Sales, label="2010", alpha=0.3)
    ax.plot(storeset_2011.WeekOfYear, storeset_2011.Weekly_Sales, label="2011", alpha=0.3)
    ax.plot(storeset_2012.WeekOfYear, storeset_2012.Weekly_Sales, label="2012", color='r')
    #ax.plot(test_pred_store.WeekOfYear, test_pred_store.Before_Weekly_Sales, label="2012-pred", color='b')
    ax.legend()
    
plt.show()

#%%
'''
# 각 Store에 대한 Weekly Sales를 시계열 그래프라고 생각하고 각 Store에 대한 Weekly Sales의 코사인 유사도를 구한 뒤에 Store끼리의 유사성을 파악
# Think of Weekly Sales for each store as a time series graph, obtain the cosine similarity of Weekly Sales for each store, and then identify the similarity between stores
def cos_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    l2_norm = (np.sqrt(sum(np.square(v1))) * np.sqrt(sum(np.square(v2))))
    similarity = dot_product / l2_norm     
    similarity = round(similarity, 3)
    return similarity

for num1 in range(1,46):
    print("\n{} Store\n".format(num1))
    scaler1 = MinMaxScaler()
    time_series1 = data[data.Store==num1]['Weekly_Sales'].values.reshape(-1, 1)
    scaled_time_series1 = scaler1.fit_transform(time_series1)
    for num2 in range(num1+1, 46) :
        time_series2 =data[data.Store==num2]['Weekly_Sales'].values.reshape(-1,1)
        scaler2 = MinMaxScaler()
        scaled_time_series2 = scaler2.fit_transform(time_series2)
        print("{} Store와 {} Store의 코사인 유사도는 {}".format(num1, num2, cos_similarity(scaled_time_series1.reshape(-1), scaled_time_series2.reshape(-1))))
'''
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
    elif prediction[i] == 1 :
        data.loc[(data.Store== i + 1), 'Type'] = 1
    else:
        data.loc[(data.Store== i + 1), 'Type'] = 2
#%%
# Analyze Correlation by Store
corr = []
for num in range(1,46):
    co = data[data.Store==num]
    co = co.reset_index()
    num_corr = co.corr()['Weekly_Sales']
    num_corr = num_corr.drop(['index', 'id','Store','Weekly_Sales'])
    corr.append(num_corr)
corr_df = pd.concat(corr, axis=1).T
corr_df.index = list(range(1,46))

f, ax = plt.subplots(figsize=(20,8))
plt.title("Correlation between point-to-point sales and variables before PROMOTION pre-processing", fontsize=15)
sns.heatmap(corr_df.T, cmap=sns.diverging_palette(240,10,as_cmap=True), ax=ax)
plt.xlabel('Store')
plt.show()
#%%

# 결측치 확인
print("Before Preprocessing\n")
print(data.isna().sum())

# 결측치 
#for i in range (1, 46):
#    data.loc[139*(i-1):91+(139*(i-1)), ['Promotion1', 'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5']] = 0


# If the value of Promotions is NAN, assume that Promotions did not proceed and replace it with 0
print("After Preprocessing (fill NAN to 0)\n")
data = data.fillna(0)
print(data.isna().sum())


#data = data.interpolate(method='values')
#print(data.isna().sum())

# Promotion 열에 꽤 많은 Nan 값이 있음 그래서 Promotion이 NAN 값인 값들을 다 삭제하기에는 너무 데이터가 줄어드는 문제가 발생


# %%
# IsHoliday Column Analysis => IsHoliday column is object so it need to change into int 
sns.barplot(x='IsHoliday', y='Weekly_Sales', data=data)

data_holiday = data.loc[data['IsHoliday']==True]
# Print when it is holiday
print(data_holiday['Date'].unique())

# Holiday
# 2010.02.12  /  2010.09.10  /  2010.11.26  /  2010.12.31
# 2011.02.11  /  2011.09.09  /  2011.11.25  /  2011.12.30
# 2012.02.10  /  2012.09.07

# There are no Holidays on the test set. period of test set is 2012.10.05 ~ 2012.10.26 (for 4 weeks and 45 store)

# Super bowl dates in train set
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


sns.barplot(x='IsHoliday', y='Weekly_Sales', data=data) # Super bowl holiday vs not-super bowl
#%%
sns.barplot(x='Labor_Day', y='Weekly_Sales', data=data) # Labor day holiday vs not-labor day
#%%
sns.barplot(x='Thanksgiving', y='Weekly_Sales', data=data) # Thanksgiving holiday vs not-thanksgiving
#%%
sns.barplot(x='Christmas', y='Weekly_Sales', data=data) # Christmas holiday vs not-Christmas
#%%

x = data['Store']
y = data['Weekly_Sales']
plt.figure(figsize=(15,5))
plt.title('Weekly Sales by Store')
plt.xlabel('Stores')
plt.ylabel('Weekly Sales')
plt.scatter(x,y)
plt.show()

plt.figure(figsize=(30,10))
fig = sns.barplot(x='Store', y='Weekly_Sales', data=data)
#%%

# Group by year
data.groupby('year')['Weekly_Sales'].mean()
# Group by month
data.groupby('month')['Weekly_Sales'].mean()
# Group by week
data.groupby('week')['Weekly_Sales'].mean()

# Visualization by month
monthly_sales = pd.pivot_table(data, values = "Weekly_Sales", columns = "year", index = "month")
monthly_sales.plot()

fig = sns.barplot(x='month', y='Weekly_Sales', data=data)


# Visualization by week
weekly_sales = pd.pivot_table(data, values = "Weekly_Sales", columns = "year", index = "week")
weekly_sales.plot()
plt.figure(figsize=(20,6))
fig = sns.barplot(x='week', y='Weekly_Sales', data=data)

# External variable check

temperature = pd.pivot_table(data, values = 'Weekly_Sales', index= "Temperature")
temperature.plot()

fuel_price = pd.pivot_table(data, values = "Weekly_Sales", index= "Fuel_Price")
fuel_price.plot()

unemployment = pd.pivot_table(data, values = "Weekly_Sales", index= "Unemployment")
unemployment.plot()
#%%
# 지점별 매출액 차이
# Differences in sales by Store

fig = plt.figure(figsize=(30,60))

for store in range(1,max(data.Store)+1):
    store_set = data[data.Store==store]
    store_set_2010 = store_set[store_set.year==2010]
    store_set_2011 = store_set[store_set.year==2011]
    store_set_2012 = store_set[store_set.year==2012]
    
    ax = fig.add_subplot(12, 4, store)
    
    plt.title(f"store_{store}")
    ax.plot(store_set_2010.week, store_set_2010.Weekly_Sales, label="2010", alpha=0.3)
    ax.plot(store_set_2011.week, store_set_2011.Weekly_Sales, label="2011", alpha=0.3)
    ax.plot(store_set_2012.week, store_set_2012.Weekly_Sales, label="2012", color='r')
    ax.legend()
    
plt.show()
#%%
# Analyze Correlation by Store
corr = []
for num in range(1,46):
    co = data[data.Store==num]
    co = co.reset_index()
    num_corr = co.corr()['Weekly_Sales']
    num_corr = num_corr.drop(['index', 'id','Store','Weekly_Sales'])
    corr.append(num_corr)
corr_df = pd.concat(corr, axis=1).T
corr_df.index = list(range(1,46))

f, ax = plt.subplots(figsize=(20,8))
plt.title("Correlation between point-to-point sales and variables After PROMOTION pre-processing", fontsize=15)
sns.heatmap(corr_df.T, cmap=sns.diverging_palette(240,10,as_cmap=True), ax=ax)
plt.xlabel('지점(Store)')
plt.show()
#%%
data['Super_Bowl'] = data['Super_Bowl'].astype(bool).astype(int) # changing T,F to 0-1
data['Thanksgiving'] = data['Thanksgiving'].astype(bool).astype(int) # changing T,F to 0-1
data['Labor_Day'] = data['Labor_Day'].astype(bool).astype(int) # changing T,F to 0-1
data['Christmas'] = data['Christmas'].astype(bool).astype(int) # changing T,F to 0-1
data['IsHoliday'] = data['IsHoliday'].astype(bool).astype(int) # changing T,F to 0-1
#%%
'''
# 전처리 이후 상관관계 분석
corr = []
for num in range(1,46):
    co = data[(data.Store==num) & (data.WeekOfYear<=43)]
    co = co.reset_index()
    num_corr = co.corr()['Weekly_Sales']
    num_corr = num_corr.drop(['index', 'id','Store','Weekly_Sales'])
    corr.append(num_corr)
corr_df = pd.concat(corr, axis=1).T
corr_df.index = list(range(1,46))

f, ax = plt.subplots(figsize=(40,15))
plt.title("Correlation after all pre-processing Before October", fontsize=15)
sns.heatmap(corr_df.T, cmap=sns.diverging_palette(240,10,as_cmap=True), ax=ax, annot=True)
plt.xlabel('Store')
plt.show()
'''
#%%
plt.figure(figsize=(28,14))
plt.xticks( fontsize=20)
plt.yticks( fontsize=20)
temp = data[data.year==2012]
sns.heatmap(data.corr(), cmap='Reds', annot=True, annot_kws={'size':12})
plt.title('Correlation Matrix only 2012', fontsize=30)
#%% 
data.set_index('Date', inplace=True) #seting date as index

plt.figure(figsize=(16,6))
data['Weekly_Sales'].plot()
plt.show()


#%%
data_week = data.resample('W').mean() #resample data as weekly

plt.figure(figsize=(16,6))
data_week['Weekly_Sales'].plot()
plt.title('Average Sales - Weekly')
plt.show()
#%%
# 2010년 데이터의 10월 이전까지
# Analyze Correlation by Store
corr = []
for num in range(1,46):
    co = data[(data.Store==num) & (data.year==2010) & (data.month <= 10)]
    co = co.reset_index()
    num_corr = co.corr()['Weekly_Sales']
    num_corr = num_corr.drop(['id','Store','Weekly_Sales'])
    corr.append(num_corr)
corr_df = pd.concat(corr, axis=1).T
corr_df.index = list(range(1,46))

f, ax = plt.subplots(figsize=(40, 15))
plt.title("Correlation 2010 Before October", fontsize=15)
sns.heatmap(corr_df.T, cmap=sns.diverging_palette(240,10,as_cmap=True), ax=ax, annot=True)
plt.xlabel('Store')
plt.show()
#%%
# 2011년 데이터의 10월 이전까지
# Analyze Correlation by Store
corr = []
for num in range(1,46):
    co = data[(data.Store==num) & (data.year==2011) & (data.month <= 10)]
    co = co.reset_index()
    num_corr = co.corr()['Weekly_Sales']
    num_corr = num_corr.drop(['id','Store','Weekly_Sales'])
    corr.append(num_corr)
corr_df = pd.concat(corr, axis=1).T
corr_df.index = list(range(1,46))

f, ax = plt.subplots(figsize=(40,15))
plt.title("Correlation 2011 Before October", fontsize=15)
sns.heatmap(corr_df.T, cmap=sns.diverging_palette(240,10,as_cmap=True), ax=ax, annot=True)
plt.xlabel('Store')
plt.show()
#%%
# 2010년 데이터의 10월 이전까지
# Analyze Correlation by Store
corr = []
for num in range(1,46):
    co = data[(data.Store==num) & (data.month <= 10)]
    co = co.reset_index()
    num_corr = co.corr()['Weekly_Sales']
    num_corr = num_corr.drop(['id','Store','Weekly_Sales'])
    corr.append(num_corr)
corr_df = pd.concat(corr, axis=1).T
corr_df.index = list(range(1,46))

f, ax = plt.subplots(figsize=(40,15))
plt.title("Correlation all year before October", fontsize=15)
sns.heatmap(corr_df.T, cmap=sns.diverging_palette(240,10,as_cmap=True), ax=ax, annot=True)
plt.xlabel('Store')
plt.show()

#%%
# 2010년 10월만 분석
# Analyze Correlation by Store
corr = []
for num in range(1,46):
    co = data[(data.Store==num) & (data.year==2010) & (data.month == 10)]
    co = co.reset_index()
    num_corr = co.corr()['Weekly_Sales']
    num_corr = num_corr.drop(['id','Store','Weekly_Sales'])
    corr.append(num_corr)
corr_df = pd.concat(corr, axis=1).T
corr_df.index = list(range(1,46))

f, ax = plt.subplots(figsize=(40,15))
plt.title("Correlation Only 2010 October", fontsize=15)
sns.heatmap(corr_df.T, cmap=sns.diverging_palette(240,10,as_cmap=True), ax=ax, annot=True)
plt.xlabel('Store')
plt.show()
#%%
# 2011년 10월만 분석
# Analyze Correlation by Store
corr = []
for num in range(1,46):
    co = data[(data.Store==num) & (data.year==2011) & (data.month == 10)]
    co = co.reset_index()
    num_corr = co.corr()['Weekly_Sales']
    num_corr = num_corr.drop(['id','Store','Weekly_Sales'])
    corr.append(num_corr)
corr_df = pd.concat(corr, axis=1).T
corr_df.index = list(range(1,46))

f, ax = plt.subplots(figsize=(40,15))
plt.title("Correlation Only 2011 October", fontsize=15)
sns.heatmap(corr_df.T, cmap=sns.diverging_palette(240,10,as_cmap=True), ax=ax, annot=True)
plt.xlabel('Store')
plt.show()
#%%
# 2012년만 분석
# Analyze Correlation by Store
corr = []
for num in range(1,46):
    co = data[(data.Store==num) & (data.year==2012)]
    co = co.reset_index()
    num_corr = co.corr()['Weekly_Sales']
    num_corr = num_corr.drop(['id','Store','Weekly_Sales'])
    corr.append(num_corr)
corr_df = pd.concat(corr, axis=1).T
corr_df.index = list(range(1,46))

f, ax = plt.subplots(figsize=(40,15))
plt.title("Correlation Only 2012", fontsize=15)
sns.heatmap(corr_df.T, cmap=sns.diverging_palette(240,10,as_cmap=True), ax=ax, annot=True)
plt.xlabel('Store')
plt.show()
#%%
#data = data.drop(['month', 'week'], axis=1)
data.to_csv("minmax_train_set.csv", index = False)
train_set = data[['Store', 'year','IsHoliday', 'Promotion3', 'WeekOfYear', 'Weekly_Sales']]

# no min max
# train_set.to_csv("minmax_train_set.csv", index = False)

# train_set.to_csv('train_set.csv', index=False) #csv파일로 생성