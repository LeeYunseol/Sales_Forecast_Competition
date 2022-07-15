import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('test.csv')
print(data.columns)

data = data.interpolate(method='values')
data = data.fillna(method='bfill')

print("결측치 수 : {}".format(data.isna().sum()))

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

data['Thanksgiving'] = data['Thanksgiving'].astype(bool).astype(int) # changing T,F to 0-1


data = data[['Fuel_Price', 'Promotion3', 'Promotion5', 'Thanksgiving']]

data_scaler = data.iloc[:, :]
print(data_scaler.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['Fuel_Price', 'Promotion3', 'Promotion5', 'Thanksgiving'])

                                         
data_df_scaled.to_csv("minmax_test_set.csv", index = False)