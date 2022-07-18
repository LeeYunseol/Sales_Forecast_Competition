import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



data_scaler = data.iloc[:, :]
print(data_scaler.columns)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_scaler)
data_scaled = scaler.transform(data_scaler)
data_df_scaled = pd.DataFrame(data=data_scaled, columns=['Promotion3', 'Promotion5', 'IsHoliday', 'week', 'month', 'Thanksgiving'])

                                         
data_df_scaled.to_csv("minmax_test_set.csv", index = False)