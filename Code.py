# Importing Necessary Libraries and Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import iplot
from matplotlib import rc
rc('font', family='AppleGothic')
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv("train.csv", encoding="utf-8")
print(data.columns)
#%%
# 데이터 정보 확인
data.info()
#%%

# 결측치 확인
print(data.isna().sum())

# 결측치 
for i in range (1, 46):
    data.loc[139*(i-1):91+(139*(i-1)), ['Promotion1', 'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5']] = 0
    
print(data.isna().sum())


data = data.interpolate(method='values')
print(data.isna().sum())

# Promotion 열에 꽤 많은 Nan 값이 있음 그래서 Promotion이 NAN 값인 값들을 다 삭제하기에는 너무 데이터가 줄어드는 문제가 발생
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

data.info()

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


sns.barplot(x='Super_Bowl', y='Weekly_Sales', data=data) # Super bowl holiday vs not-super bowl
sns.barplot(x='Labor_Day', y='Weekly_Sales', data=data) # Labor day holiday vs not-labor day
sns.barplot(x='Thanksgiving', y='Weekly_Sales', data=data) # Thanksgiving holiday vs not-thanksgiving
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
# Change type of 'Date' : object to datedime
data['Date'] = pd.to_datetime(data["Date"])
data['week'] =data['Date'].dt.week
data['month'] =data['Date'].dt.month 
data['year'] =data['Date'].dt.year

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
plt.title("지점별 매출액과 변수들간의 상관관계", fontsize=15)
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
# 전처리 이후 상관관계 분석
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
plt.title("지점별 매출액과 변수들간의 상관관계", fontsize=15)
sns.heatmap(corr_df.T, cmap=sns.diverging_palette(240,10,as_cmap=True), ax=ax)
plt.xlabel('지점(Store)')
plt.show()
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
print(data_week.columns)
train_set = data_week[['Fuel_Price', 'Promotion3', 'Promotion5', 'Thanksgiving', 'Weekly_Sales']]

train_set.to_csv('train_set.csv', index=False) #csv파일로 생성