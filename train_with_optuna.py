import xgboost as xgb
#from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
import optuna
from matplotlib import pyplot
import matplotlib.pyplot as plt

train_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/minmax_train_set.csv')
test_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/minmax_test_set.csv')

original_train_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/train_set.csv')
original_test_df = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/test_set.csv')

scaler_for_weekly_sales = MinMaxScaler()
scaler_for_weekly_sales.fit(train_df['Original_Weekly_Sales'].values.reshape(-1, 1))

# Feautre for training
feature = list(train_df.columns)
feature.remove('Weekly_Sales')
feature.remove('Original_Weekly_Sales')
#feature.remove('day')
#feature.remove('Type')

# X_train
temp1 = train_df[train_df.year_2012!=1]
temp2 =  train_df[((train_df.year_2012==1) & (train_df.month_9 != 1))]
X_train = pd.concat([temp1, temp2])
y_train = X_train.Weekly_Sales

X_train = X_train[feature]

X_test = train_df[(train_df.year_2012==1) & (train_df.month_9 == 1)]
y_test = X_test.Weekly_Sales
X_test = X_test[feature]



def objective(trial, X_train, y_train, X_test, y_test):
    
    param = {
  # this parameter means using the GPU when training our model to speedup the training process
        'tree_method' : 'gpu_hist',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.8,0.9,1.0]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 1),
        'n_estimators': trial.suggest_int('n_estimators', 5000, 20000),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'random_state': trial.suggest_categorical('random_state', [2022]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    }
    model = xgb.XGBRegressor(**param)  
    
    model.fit(X_train, y_train ,eval_set=[(X_test, y_test)],early_stopping_rounds=100,verbose=False)
    
    preds = model.predict(X_test)
    
    rmse = mean_squared_error(y_test, preds,squared=False)
    
    return rmse


#%%
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial : objective(trial, X_train, y_train, X_test, y_test), n_trials=30)
print("\nDone!")
print('Number of finished trials:', len(study.trials))
print("Best Score : {}".format(study.best_trial.value))
print('Best trial:', study.best_trial.params)
#%%
df = study.trials_dataframe()
#%%
#plot_optimization_histor: shows the scores from all trials as well as the best score so far at each point.
#optuna.visualization.plot_optimization_history(study)
#%%
#Visualize parameter importances.
#optuna.visualization.plot_param_importances(study)
#%%
#Visualize empirical distribution function
#optuna.visualization.plot_edf(study)
#%%
# Let's create an XGBoostRegressor model with the best hyperparameters

Best_trial = study.best_trial.params
Best_trial

#%%

preds = np.zeros(test_df.shape[0])
kf = KFold(n_splits=5,random_state=2022,shuffle=True)
rmse=[]  # list contains rmse for each fold
n=0

X_train = train_df[feature]
y_train = train_df.Weekly_Sales



for trn_idx, test_idx in kf.split(X_train[feature], y_train):
    
    X_tr, X_val = X_train[feature].iloc[trn_idx], X_train[feature].iloc[test_idx]
    
    y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[test_idx]
    
    model = xgb.XGBRegressor(**Best_trial)
    model.fit(X_tr, y_tr, verbose=False)
    preds+=model.predict(test_df[feature])/kf.n_splits
    rmse.append(mean_squared_error(y_val, model.predict(X_val), squared=False))
    print(f"fold: {n+1} ==> rmse: {rmse[n]}")
    n+=1
print(np.mean(rmse))
preds = scaler_for_weekly_sales.inverse_transform(preds.reshape(-1, 1))
preds = np.expm1(preds)
test_df["Weekly_Sales"] = preds
original_test_df["Weekly_Sales"] = preds
#%%

fig = plt.figure(figsize=(30,60))
plt.title(feature)
for store in range(1,46):
    storeset = original_train_df[original_train_df.Store==store]
    storeset_2010 = storeset[(storeset.year==2010) & (storeset.month<=10)]
    storeset_2011 = storeset[(storeset.year==2011) & (storeset.month<=10)]
    storeset_2012 = storeset[(storeset.year==2012) & (storeset.month<=9)]
    
    test_pred_store = original_test_df[original_test_df.Store==store]
    
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
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(40, 20))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X_train[feature].columns)[sorted_idx])
plt.title('Feature Importance')
#%%
sample_submission = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/sample_submission.csv')
sample_submission["Weekly_Sales"] = test_df.Weekly_Sales
sample_submission.to_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/submission.csv',index = False)

    