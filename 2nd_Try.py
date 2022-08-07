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
import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/train_data/minmax_train_set.csv')
test_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/prepossed_test_data/minmax_test_set.csv')
original_train_df = pd.read_csv('train.csv')


scaler_for_weekly_sales = MinMaxScaler()
scaler_for_weekly_sales.fit(original_train_df['Weekly_Sales'].values.reshape(-1, 1))
#%%
'''
feature = corr_df.iloc[0].loc[corr_df.iloc[0] > 0.2]
pratice = feature.index
pratice2 = train_df[pratice]
'''
#%%
models = []
score_list = []
rmse_list = []
rate_list = []
feature_list = []
# 꼭 있어야하는 정보 
features = ['IsHoliday', 'year','WeekOfYear', 'month', 'day']

corr = []
for num in range(1,46):
    co = train_df[(train_df.Store==num) & (train_df.year == 2012) & (train_df.month <= 10)]
    co = co.reset_index()
    num_corr = co.corr()['Weekly_Sales']
    num_corr = num_corr.drop(['index', 'id','Store','Weekly_Sales'])
    corr.append(num_corr)
corr_df = pd.concat(corr, axis=1).T
corr_df.index = list(range(1,46))

# 일단 내가 중요하다고 생각하는 데이터 셋 하나 
feature1 = ['IsHoliday', 'year','WeekOfYear', 'month', 'day']


# for문 돌리면서 특정 store의 최적의 모델과 하이퍼파라미터 데이터 set 찾기
# 평가 지표는 2012년 9월 예측 값을 기준으로
for store in tqdm(range(1, 46)):
    # feature2 subset 만들기
    feature = corr_df.iloc[store-1].loc[corr_df.iloc[store-1] >= 0.1]
    feature2 = list(feature.index)
    '''
    # 다중공선성 고려 : IsHoliday / SuperBowl, Laborday(WeekOfYear과 week, month는 이미 지움)
    if ('IsHoliday' in feature2) and ('Thanksgiving' in feature2) :
        if feature['IsHoliday'] >= feature['Thanksgiving'] :
            feature2.remove('Thanksgiving')
        else :
            feature2.remove('IsHoliday')
            
    if ('IsHoliday' in feature2) and ('Super_Bowl' in feature2) :
        if feature['IsHoliday'] >= feature['Super_Bowl'] :
            feature2.remove('Super_Bowl')
        else :
            feature2.remove('IsHoliday')
    '''        
    temp = ['IsHoliday', 'year','WeekOfYear', 'month', 'day']
    temp.extend(feature.index)
    feature2_2 = list(set(list(temp)))
    '''
    if ('IsHoliday' in feature2_2) and ('Thanksgiving' in feature2_2) :
        if feature['IsHoliday'] >= feature['Thanksgiving'] :
            feature2_2.remove('Thanksgiving')
        else :
            feature2_2.remove('IsHoliday')
            feature2_2
    if ('IsHoliday' in feature2_2) and ('Super_Bowl' in feature2_2) :
        if feature['IsHoliday'] >= feature['Super_Bowl'] :
            feature2_2.remove('Super_Bowl')
        else :
            feature2_2.remove('IsHoliday')
    '''        
    # feature3 subset 만들기
    feature = corr_df.iloc[store-1].loc[corr_df.iloc[store-1] >= 0.2]
    feature3 = list(feature.index)
    
    '''
    # 다중공선성 고려 : IsHoliday / SuperBowl, Laborday(WeekOfYear과 week, month는 이미 지움)
    if ('IsHoliday' in feature3) and ('Thanksgiving' in feature3) :
        if feature['IsHoliday'] >= feature['Thanksgiving'] :
            feature3.remove('Thanksgiving')
        else :
            feature3.remove('IsHoliday')
            
    if ('IsHoliday' in feature3) and ('Super_Bowl' in feature3) :
        if feature['IsHoliday'] >= feature['Super_Bowl'] :
            feature3.remove('Super_Bowl')
        else :
            feature3.remove('IsHoliday')
    '''
    temp = ['IsHoliday', 'year','WeekOfYear', 'month', 'day']
    temp.extend(feature.index)
    feature3_2 = list(set(list(temp)))
    '''
    if ('IsHoliday' in feature3_2) and ('Thanksgiving' in feature3_2) :
        if feature['IsHoliday'] >= feature['Thanksgiving'] :
            feature3_2.remove('Thanksgiving')
        else :
            feature3_2.remove('IsHoliday')
            
    if ('IsHoliday' in feature3_2) and ('Super_Bowl' in feature3_2) :
        if feature['IsHoliday'] >= feature['Super_Bowl'] :
            feature3_2.remove('Super_Bowl')
        else :
            feature3_2.remove('IsHoliday')
   '''         
    # feature4 subset 만들기
    feature = corr_df.iloc[store-1].loc[corr_df.iloc[store-1] >= 0.4]
    feature4 = list(feature.index)
    '''
    # 다중공선성 고려 : IsHoliday / SuperBowl, Laborday(WeekOfYear과 week, month는 이미 지움)
    if ('IsHoliday' in feature4) and ('Thanksgiving' in feature4) :
        if feature['IsHoliday'] >= feature['Thanksgiving'] :
            feature4.remove('Thanksgiving')
        else :
            feature4.remove('IsHoliday')
            
    if ('IsHoliday' in feature4) and ('Super_Bowl' in feature4) :
        if feature['IsHoliday'] >= feature['Super_Bowl'] :
            feature4.remove('Super_Bowl')
        else :
            feature4.remove('IsHoliday')
    '''
    temp = ['IsHoliday', 'year','WeekOfYear', 'month', 'day']
    temp.extend(feature.index)
    feature4_2 = list(set(list(temp)))
    '''
    if ('IsHoliday' in feature4_2) and ('Thanksgiving' in feature4_2) :
        if feature['IsHoliday'] >= feature['Thanksgiving'] :
            feature4_2.remove('Thanksgiving')
        else :
            feature4_2.remove('IsHoliday')
            
    if ('IsHoliday' in feature4_2) and ('Super_Bowl' in feature4_2) :
        if feature['IsHoliday'] >= feature['Super_Bowl'] :
            feature4_2.remove('Super_Bowl')
        else :
            feature4_2.remove('IsHoliday')
      '''      
    
    train_store = train_df[train_df.Store==store]
    
    # 2010, 2011, 2012 년도 별로 데이터 분리
    # 2012-09에 대해 예측하려고 하기 때문에 2012년도는 9월을 포함하지 않음
    train_store_2010 = train_store[(train_store.year==2010) & (train_store.WeekOfYear<=38)]
    train_store_2011 = train_store[(train_store.year==2011) & (train_store.WeekOfYear<=39)]
    train_store_2012 = train_store[(train_store.year==2012) & (train_store.WeekOfYear<=35)]
    train_store_2010_2011 = pd.concat([train_store_2010, train_store_2011])
    train_store_2010_2012 = pd.concat([train_store_2010, train_store_2012])
    train_store_2011_2012 = pd.concat([train_store_2011, train_store_2012])
    train_store_all = pd.concat([train_store_2010, train_store_2011, train_store_2012])
    
    # 2012년도 9월에 대해서 예측
    X_test = train_store[(train_store.year==2012) & (train_store.WeekOfYear <= 39) & (36<=train_store.WeekOfYear)]
    
    # 각각의 모델 GridSearch   
    train_list = [train_store_2010, train_store_2011, train_store_2012,
                  train_store_2010_2011, train_store_2010_2012, train_store_2011_2012, train_store_all]
    
    rmse_min = 1000000
    score_max = 0
    for i in tqdm(range(len(train_list))) :

        max_depth_list = [3, 5, 10]
        n_estimator_list = [100, 200, 300, 400]
        lr_list =[0.05, 0.1]

        #rate = 0.1
        #best_rate = 0.1
        for depth in max_depth_list :
            for rate in lr_list :
                for n_estimator in n_estimator_list :
                    for feature in [feature1, feature2, feature3, feature4, feature2_2, feature3_2, feature4_2] : 
                        if len(feature) == 0 :
                            continue
                            
                        model = XGBRegressor(colsample_bytree=0.8, max_depth= depth, learning_rate= rate, n_estimators=n_estimator,
                                           random_state =2022, nthread = -1, n_jobs=-1)
                        ''' 
                        model2 = ExtraTreesRegressor(bootstrap=False, criterion="squared_error", max_depth=depth,
                                                     max_features="auto", max_leaf_nodes=None,
                                                     min_impurity_decrease=0.0,
                                                     min_samples_leaf=2, min_samples_split=5,
                                                     min_weight_fraction_leaf=0.0, n_estimators=n_estimator, n_jobs=30,
                                                     oob_score=False, random_state=2022, warm_start=False)
                        
                        model3 = RandomForestRegressor(n_estimators=n_estimator, max_depth = depth)
                        
                        for model in [model1, model2, model3]:
                        '''
                        
                        model.fit(train_list[i][feature], train_list[i].Weekly_Sales,
                                  eval_set=[(X_test[feature], X_test.Weekly_Sales)],
                                  eval_metric='rmse', 
                                  early_stopping_rounds=20,
                                  verbose = 0)
                        y_pred = model.predict(X_test[feature])
                        RMSE = np.sqrt(metrics.mean_squared_error(X_test.Weekly_Sales, y_pred))
                        score = r2_score(X_test.Weekly_Sales, y_pred)
                        #print("Store {}  MODEL : {} feature set : {}  depth : {}, n_estimator : {} learning rate : {}".format(store, model, feature,depth, n_estimator, rate))
                        # print("Store {}".format(store))
                        # print("RMSE : {}".format(RMSE))
                        # print("R^2:", score)
                        if (RMSE < rmse_min) : #and (score > 0) and (score > score_max):
                            rmse_min = RMSE
                            score_max = score
                            best_depth = depth
                            best_n_estimator = n_estimator
                            best_train_data_name = i
                            best_train_data = train_list[i]
                            best_rate = rate
                            best_feature = feature
                            best_model = "xgboost"
                            '''
                            if model == model1 :
                                best_model = "xgboost"
                            elif model == model2 :
                                best_model = "extratree"
                                
                            else :
                                best_model = "randomforest"
                            '''
                        
    # 전체 데이터를 학습하기 전에 2012년 9월에 대한 rmse와 r^2 스코어 확인
    if best_model == "xgboost" :
        check_model = XGBRegressor(colsample_bytree=0.8, max_depth= best_depth, learning_rate= best_rate, n_estimators=best_n_estimator,
                           random_state =2022, nthread = -1, n_jobs=-1) 
        check_model.fit(best_train_data[best_feature], best_train_data.Weekly_Sales,
                  eval_set=[(X_test[best_feature], X_test.Weekly_Sales)],
                  eval_metric='rmse', 
                  early_stopping_rounds=20,
                  verbose = 0)
        y_pred = check_model.predict(X_test[best_feature])
        
    elif best_model == "extratree" :
        check_model = ExtraTreesRegressor(bootstrap=False, criterion="squared_error", max_depth=best_depth,
                                     max_features="auto", max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     min_samples_leaf=2, min_samples_split=5,
                                     min_weight_fraction_leaf=0.0, n_estimators=best_n_estimator, n_jobs=30,
                                     oob_score=False, random_state=2022, warm_start=False)
        check_model.fit(best_train_data[best_feature], best_train_data.Weekly_Sales)
        y_pred = check_model.predict(X_test[best_feature])
        
    else  :
        check_model = RandomForestRegressor(n_estimators=best_n_estimator, max_depth = best_depth)
        check_model.fit(best_train_data[best_feature], best_train_data.Weekly_Sales)
        y_pred = check_model.predict(X_test[best_feature])
        
    print("\nStore {}\n".format(store))
    x = np.sqrt(metrics.mean_squared_error(X_test.Weekly_Sales, y_pred))
    print("RMSE: ", x) #RMSE
    score = r2_score(X_test.Weekly_Sales, y_pred)
    print("R^2:", score)
    print("Model : {}".format(best_model))
    score_list.append(score)
    rmse_list.append(x)
    feature_list.append(best_feature)
    rate_list.append(best_rate)
    
    #  최적의 학습 데이터와 파라미터를 찾고 난 이후 전체 데이터로 학습 (10월까지)
    if best_train_data_name == 0 :
        last_train_data = train_store[(train_store.year==2010) & (train_store.WeekOfYear<=43)]
        
    if best_train_data_name == 1 :
        last_train_data = train_store[(train_store.year==2011) & (train_store.WeekOfYear<=43)]
        
    if best_train_data_name == 2 :
        last_train_data = train_store[(train_store.year==2012) & (train_store.WeekOfYear<=39)]
        
    if best_train_data_name == 3 :
        temp1 = train_store[(train_store.year==2010) & (train_store.WeekOfYear<=43)]
        temp2 = train_store[(train_store.year==2011) & (train_store.WeekOfYear<=43)]
        last_train_data = pd.concat([temp1, temp2])
    
    if best_train_data_name == 4 :
        temp1 = train_store[(train_store.year==2010) & (train_store.WeekOfYear<=43)]
        temp2 = train_store[(train_store.year==2012) & (train_store.WeekOfYear<=39)]
        last_train_data = pd.concat([temp1, temp2])
        
    if best_train_data_name == 5 :
        temp1 = train_store[(train_store.year==2011) & (train_store.WeekOfYear<=43)]
        temp2 = train_store[(train_store.year==2012) & (train_store.WeekOfYear<=39)]
        last_train_data = pd.concat([temp1, temp2])
            
    if best_train_data_name == 6 :
        temp1 = train_store[(train_store.year==2010) & (train_store.WeekOfYear<=43)]
        temp2 = train_store[(train_store.year==2011) & (train_store.WeekOfYear<=43)]
        temp3 = train_store[(train_store.year==2012) & (train_store.WeekOfYear<=39)]
        last_train_data = pd.concat([temp1, temp2, temp3])        
    
    
    if best_model == "xgboost" :
        last_model = XGBRegressor(colsample_bytree=0.8, max_depth= best_depth, learning_rate= best_rate, n_estimators=best_n_estimator,
                           random_state =2022, nthread = -1, n_jobs=-1) 
        
    elif best_model == "extratree" :
        last_model = ExtraTreesRegressor(bootstrap=False, criterion="squared_error", max_depth=best_depth,
                                     max_features="auto", max_leaf_nodes=None,
                                     min_impurity_decrease=0.0,
                                     min_samples_leaf=2, min_samples_split=5,
                                     min_weight_fraction_leaf=0.0, n_estimators=best_n_estimator, n_jobs=30,
                                     oob_score=False, random_state=2022, warm_start=False)
    
    else:
        last_model = RandomForestRegressor(n_estimators=best_n_estimator, max_depth = best_depth)
    
    last_model.fit(last_train_data[best_feature], last_train_data.Weekly_Sales,  
                   eval_set = [(last_train_data[best_feature], last_train_data.Weekly_Sales)],
                   eval_metric='rmse',
                   early_stopping_rounds=20,
                   verbose = 0
                   )
    
    models.append(last_model)
#%%

for i in range(45) :
    print("Store {}".format(i+1))
    print("BEST FEATURE SET : {}  MODEL : {} RMSE : {} R^2 : {} lr : {}\n".format(feature_list[i], models[i], rmse_list[i], score_list[i], rate_list[i]))
    
# R^2가 음수인 부분은 재학습

#%%

# 모델 저장 
from keras.models import load_model
import tensorflow as tf
index = 1
for model in models :
    model.save_model('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/models/Store{}'.format(index))
    index+=1

#%%
# 학습 종료 후 test.csv 평가

pred = []
for store in range(1, 46):
    test_store = test_df[test_df.Store==store]
    prediction = models[store-1].predict(test_store[feature_list[store-1]])
    # prediction = scaler_for_weekly_sales.inverse_transform(prediction.reshape(-1, 1))
    pred += prediction.tolist()

test_pred = test_df.copy()
test_pred["Weekly_Sales"] = pred
#%%
fig = plt.figure(figsize=(30,60))

for store in range(1,46):
    storeset = train_df[train_df.Store==store]
    storeset_2010 = storeset[(storeset.year==2010) & (storeset.WeekOfYear<=43)]
    storeset_2011 = storeset[(storeset.year==2011) & (storeset.WeekOfYear<=43)]
    storeset_2012 = storeset[(storeset.year==2012) & (storeset.WeekOfYear<=43)]
    
    test_pred_store = test_pred[test_pred.Store==store]
    
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
#%%
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission["Weekly_Sales"] = test_pred.Weekly_Sales
sample_submission.to_csv('submission.csv',index = False)
sample_submission

    