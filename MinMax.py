import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('train_set.csv')
print(data.columns)
#%%
data_scaler = data.iloc[:, :]
print(data_scaler.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['Fuel_Price', 'Promotion3', 'Promotion5', 'Thanksgiving',
       'target'])



data_df_scaled['Original Target'] = data['Weekly_Sales']
                                         
data_df_scaled.to_csv("minmax_train_set.csv", index = False)

