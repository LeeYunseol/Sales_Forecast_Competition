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
train_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/train_data/minmax_train_set.csv')
test_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/prepossed_test_data/minmax_test_set.csv')
original_train_df = pd.read_csv('train.csv')


scaler_for_weekly_sales = MinMaxScaler()
scaler_for_weekly_sales.fit(original_train_df['Weekly_Sales'].values.reshape(-1, 1))

models = []
score_list = []
rmse_list = []
features = ['Store', 'year','Thanksgiving', 'Promotion3','WeekOfYear']

# 사용 모델 : Xgboost
for store in tqdm(range(1, 46)):
    train_store = train_df[train_df.Store==store]
    
    # 2010, 2011, 2012 년도 별로 데이터 분리
    # 2012-09에 대해 예측하려고 하기 때문에 2012년도는 9월을 포함하지 않음
    train_store_2010 = train_store[(train_store.year==2010) & (train_store.WeekOfYear<=38)]
    train_store_2011 = train_store[(train_store.year==2011) & (train_store.WeekOfYear<=38)]
    train_store_2012 = train_store[(train_store.year==2012) & (train_store.WeekOfYear<=34)]
    train_store_2010_2011 = pd.concat([train_store_2010, train_store_2011])
    train_store_2010_2012 = pd.concat([train_store_2010, train_store_2012])
    train_store_2011_2012 = pd.concat([train_store_2011, train_store_2012])
    train_store_all = pd.concat([train_store_2010, train_store_2011, train_store_2012])
    
    # 2012년도 9월에 대해서 예측
    X_test = train_store[(train_store.year==2012) & (train_store.WeekOfYear <= 38) & (35<=train_store.WeekOfYear)]
    
    # 각각의 모델 GridSearch   
    train_list = [train_store_2010, train_store_2011, train_store_2012,
                  train_store_2010_2011, train_store_2010_2012, train_store_2011_2012, train_store_all]
    
    rmse_min = 1000000
    
    for i in tqdm(range(len(train_list))) :

        max_depth_list = [3, 5, 10]
        n_estimator_list = [30, 50, 100, 200]
        lr_list = [0.1, 0.2, 0.3]
        #rate = 0.1
        #best_rate = 0.1
        for depth in max_depth_list :
            for rate in lr_list :
                for n_estimator in n_estimator_list :
                    model = XGBRegressor(colsample_bytree=0.8, max_depth= depth, learning_rate= rate, n_estimators=n_estimator,
                                       random_state =2022, nthread = -1, n_jobs=-1)
                        
                    model.fit(train_list[i][features], train_list[i].Weekly_Sales, eval_metric='rmse')
                    y_pred = model.predict(X_test[features])
                    RMSE = np.sqrt(metrics.mean_squared_error(X_test.Weekly_Sales, y_pred))
                    score = r2_score(X_test.Weekly_Sales, y_pred)
                    print("Store {} depth : {}, n_estimator : {}".format(store, depth, n_estimator))
                    print("RMSE : {}".format(RMSE))
                    print("R^2:", score)
                    if (RMSE < rmse_min) : # (score > 0) :
                        rmse_min = RMSE
                        best_depth = depth
                        best_n_estimator = n_estimator
                        best_train_data_name = i
                        best_train_data = train_list[i]
                        best_rate = rate
                        
                    
    # 전체 데이터를 학습하기 전에 rmse와 r^2 스코어 확인
    model = XGBRegressor(colsample_bytree=0.8, max_depth= best_depth, learning_rate= best_rate, n_estimators=best_n_estimator,
                       random_state =2022, nthread = -1, n_jobs=-1)
    model.fit(best_train_data[features], best_train_data.Weekly_Sales, eval_metric='rmse')
    y_pred = model.predict(X_test[features])
    
    print("\nStore {}\n".format(store))
    x = np.sqrt(metrics.mean_squared_error(X_test.Weekly_Sales, y_pred))
    print("RMSE: ", x) #RMSE
    score = r2_score(X_test.Weekly_Sales, y_pred)
    print("R^2:", score)
    score_list.append(score)
    rmse_list.append(x)
    
    #  최적의 학습 데이터와 파라미터를 찾고 난 이후 전체 데이터로 학습 (10월까지)
    if best_train_data_name == 0 :
        last_train_data = train_store[(train_store.year==2010) & (train_store.WeekOfYear<=43)]
        
    if best_train_data_name == 1 :
        last_train_data = train_store[(train_store.year==2011) & (train_store.WeekOfYear<=43)]
        
    if best_train_data_name == 2 :
        last_train_data = train_store[(train_store.year==2012) & (train_store.WeekOfYear<=38)]
        
    if best_train_data_name == 3 :
        temp1 = train_store[(train_store.year==2010) & (train_store.WeekOfYear<=43)]
        temp2 = train_store[(train_store.year==2011) & (train_store.WeekOfYear<=43)]
        last_train_data = pd.concat([temp1, temp2])
    
    if best_train_data_name == 4 :
        temp1 = train_store[(train_store.year==2010) & (train_store.WeekOfYear<=43)]
        temp2 = train_store[(train_store.year==2012) & (train_store.WeekOfYear<=38)]
        last_train_data = pd.concat([temp1, temp2])
        
    if best_train_data_name == 5 :
        temp1 = train_store[(train_store.year==2011) & (train_store.WeekOfYear<=43)]
        temp2 = train_store[(train_store.year==2012) & (train_store.WeekOfYear<=38)]
        last_train_data = pd.concat([temp1, temp2])
            
    if best_train_data_name == 6 :
        temp1 = train_store[(train_store.year==2010) & (train_store.WeekOfYear<=43)]
        temp2 = train_store[(train_store.year==2011) & (train_store.WeekOfYear<=43)]
        temp3 = train_store[(train_store.year==2012) & (train_store.WeekOfYear<=38)]
        last_train_data = pd.concat([temp1, temp2, temp3])        
            
            
            
    model = XGBRegressor(colsample_bytree=0.8, max_depth= best_depth, learning_rate= best_rate, n_estimators=best_n_estimator,
                       random_state =2022, nthread = -1, n_jobs=-1)
    model.fit(last_train_data[features], last_train_data.Weekly_Sales, eval_metric='rmse')
    
    models.append(model)
#%%

for i in range(45) :
    print("Store {}".format(i+1))
    print("RMSE : {} R^2 : {}\n".format(rmse_list[i], score_list[i]))
    
# R^2가 음수인 부분은 재학습

#%%
# 학습 종료 후 test.csv 평가

pred = []
for store in range(1, 46):
    test_store = test_df[test_df.Store==store]
    
    prediction = models[store-1].predict(test_store[features])
    # prediction = scaler_for_weekly_sales.inverse_transform(prediction.reshape(-1, 1))
    pred += prediction.tolist()

test_pred = test_df.copy()
test_pred["Weekly_Sales"] = pred

#%%
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission["Weekly_Sales"] = test_pred.Weekly_Sales
sample_submission.to_csv('submission.csv',index = False)
sample_submission

    