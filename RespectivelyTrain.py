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

features = ['Store', 'year','Thanksgiving', 'Promotion3','WeekOfYear']

# 사용 모델 : Xgboost
for store in tqdm(range(1, 46)):
    train_store = train_df[train_df.Store==store]
    
    # 2010, 2011, 2012 년도 별로 데이터 분리
    # 2012-09에 대해 예측하려고 하기 때문에 2012년도는 9월을 포함하지 않음
    train_store_2010 = train_store[(train_store.year==2010) & (train_store.WeekOfYear<=43)]
    train_store_2011 = train_store[(train_store.year==2011) & (train_store.WeekOfYear<=43)]
    train_store_2012 = train_store[(train_store.year==2012) & (train_store.WeekOfYear<=39)]
    train_store_2010_2011 = pd.concat([train_store_2010, train_store_2011])
    train_store_2010_2012 = pd.concat([train_store_2010, train_store_2012])
    train_store_2011_2012 = pd.concat([train_store_2011, train_store_2012])
    train_store_all = pd.concat([train_store_2010, train_store_2011, train_store_2012])
    # 2012년도 9월에 대해서 예측
    # X_test = train_store[(train_store.Year==2012) & (train_store.Month==9)]
    
    # 각각의 모델 GridSearch   
    train_list = [train_store_2010, train_store_2011, train_store_2012,
                  train_store_2010_2011, train_store_2010_2012, train_store_2011_2012, train_store_all]
    
    rmse_min = 1000000
    
    for train_data in tqdm(train_list) :
        model = XGBRegressor()
        parameters = {
            'learning_rate': [0.1, 0.2, 0.35, 0.5], #so called `eta` value
            'max_depth': [3, 5, 10],
            'min_child_weight': [4],
            'silent': [1],
            'subsample': [0.7],
            'colsample_bytree': [0.7, 0.8],
                      'n_estimators': [30, 50, 100, 200],
                      'random_state' :[2022]
                     }

        xgb_grid = GridSearchCV(model,
                                parameters,
                                cv = 5,
                                scoring ='neg_mean_absolute_error',
                                n_jobs = 5,
                                verbose=True)
        xgb_grid.fit(train_data[features], train_data.Weekly_Sales)
        
        if xgb_grid.best_score_ < rmse_min :
            rmse_min = xgb_grid.best_score_
            best_parameter = xgb_grid.best_params_
            best_train_data = train_data
    
    # 그리드 서치 종료
    model=xgb_grid.best_estimator_
    model.fit(best_train_data[features], best_train_data["Weekly_Sales"])
    models.append(model)
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

    