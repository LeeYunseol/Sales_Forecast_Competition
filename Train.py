# Import important libraries
import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision #
import torchvision.datasets as dset
import torchvision.transforms as tr 
from torch.utils.data import TensorDataset,DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import src.preprocess as pp
import src.model as md
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable 
import matplotlib.pyplot as plt
import seaborn as sns
import os 

#%%
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print("Device Chekck : {}".format(device))

path = 'C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/train_data'
file_list = os.listdir(path)

# Load Data
data={}
for i in file_list: 
    f_name = 'prepossed train set'
    df = pd.read_csv(path+'/'+i)
    data[f_name] = df.iloc[:,:]
    
# Train hyper parameter
# We dont need train_split because we have few of datas and we have another test set provieded by DACON
test_rate = 0.2
hid_dim = 512
num_epoch = 3400

error = {}
for z in [1]:
    w_size = z # window size
    p_size = z # predict size
    out_dim = p_size # output dim
    

    #시계열 데이터 생성 
    ts_data = {}
    
    for i in data:
        torch.manual_seed(2022)
        
        X = data[i]
        real = X[['Original Target']]
        # Feautres / target(MinMaxScaler Weekly_Sales) / Original Target(Original Weekly_Sales)
        X = X.iloc[:,:-2]
        
        #정답 스케일링 
        scaler = MinMaxScaler()
        real_scaled = pd.DataFrame(scaler.fit_transform(real), columns = ['target']) 
        y=real_scaled
        
        
        # Make time-series data
        X_s, y_s = pp.make_sequence_train_dataset(X, y, w_size, p_size)
        ts_data[i] = [X_s, y_s]
    
        idx = int(test_rate*len(X_s))
        X_train = X_s[:-idx]
        y_train = y_s[:-idx]
        
        X_test = X_s[-idx:]
        y_test = y_s[-idx:]
        
        #텐서 생성 
        
        train_data = pp.TensorData(X_train, y_train)
        train_loader = DataLoader(train_data, batch_size = 32, shuffle=False)
        

        test_data = pp.TensorData(X_test, y_test)
        test_loader = DataLoader(test_data, batch_size = y_test.shape[0], shuffle=False)
        
        #입력차원 설정 
        in_dim = X_train.shape[-1]
  
        
        #모델 구축 
        m = md.LSTM(input_size = in_dim, hidden_size = hid_dim, out_dim = out_dim, num_layers = 1, seq_length = w_size).to(device)
        
        # MSE 
        crit = torch.nn.MSELoss()
        para = list(m.parameters())
        # First learning rate was 0.00001
        optimizer = optim.Adam(para, 0.001) # lstm 0.001 epoch 600일대 괜찮았음 -> 하지만 더 학습을 하면 난리가 남
        #LSTM 0.0001 3400 530000
        
        final_test_rmse = 10e+10
        final_test_mape = 0
        
        total_test_rmse = []
        total_test_mape = []  
        
        # Train
        m.train()
        errors = []
        for e in range(1,num_epoch+1):
            
            #배치별 훈련
            
            y_trains = []
            outs = []
            
            #배치 학습     
            train_rmse = []
            train_mape = []
            
            for _, samples in enumerate(train_loader): 
                X_train, y_train = samples
                
                optimizer.zero_grad()
                train_out = m(X_train.to(device))
                y_train = y_train.reshape(-1,p_size).to(device)
            
                #배치별 업데이트
                loss = crit(train_out, y_train)
                
                loss.backward()
                optimizer.step()
                
                train_out = train_out.detach().cpu().numpy()
                y_train = y_train.detach().cpu().numpy()
                
                #역스케일링 
                for j in range(p_size):
                    train_out[:,[j]] = scaler.inverse_transform(train_out[:,[j]])
                    y_train[:,[j]] = scaler.inverse_transform(y_train[:,[j]])
                
                outs.extend(train_out)
                y_trains.extend(y_train)
                
            
            outs = np.array(outs)
            y_trains = np.array(y_trains)
                
            #마지막 값들만 시각화 위해서 저장 
            result_train = pd.DataFrame({'real':y_trains[:,-1], 'predict':outs[:,-1]})
            
            result_train = result_train.reset_index()
            
            rmse = (mse(outs, y_trains)**0.5)
            mape = (np.mean(abs((np.array(y_trains)-np.array(outs))/np.array(y_trains)))*100)
            
            train_rmse.append(rmse)
            train_mape.append(mape)
            
            #테스트 
            m.eval()
            with torch.no_grad():
                for idx, samples in enumerate(test_loader):
                    X_test, y_test = samples
                    #X_test = X_test.to(device)
                    #y_test = y_test.to(device)
                    
                    
                    out = m(X_test.to(device))
                    y_test = y_test.reshape(-1,p_size).to(device)
                    
                    #테스트 출력 값  
                    out = out.detach().cpu().numpy()
                    
                    y_test = y_test.detach().cpu().numpy()
                    
                    
                    #역스케일링 
                    for j in range(p_size):
                            out[:,[j]] = scaler.inverse_transform(out[:,[j]])
                            y_test[:,[j]] = scaler.inverse_transform(y_test[:,[j]])
                        
                    test_rmse = (mse(out, y_test)**0.5)
                    test_mape = (np.mean(abs((np.array(y_test)-np.array(out))/np.array(y_test)))*100)
                    
                    #마지막 값들만 시각화 위해서 저장 
                    result_test = pd.DataFrame({'real':y_test[:,-1], 'predict':out[:,-1]})
                    result_test = result_test.reset_index()
                    
                    result = pd.concat([result_train, result_test], ignore_index=True)
                    result = result.drop('index', axis=1)
                    result = result.reset_index()
                    
                    #테스트 결과 시각화 
                    if e % 10 ==0 :
                        
                        fig, ax = plt.subplots(1,1, figsize=(10,7))
                        sns.lineplot(data = result, x='index', y='real',color ='red', label = 'real')
                        sns.lineplot(data = result, x='index', y='predict',color ='blue', label ='predict')
                        plt.axvline(x =len(result_train) , color = "red", linestyle='dashed')
                        ax.set(title = 'LSTM,  data set: '+i+ '\ntrain rmse: '+str(np.mean(train_rmse).round(3))+', train mape: '+str(np.mean(train_mape).round(3))+
                    
                               '\n test rmse: '+str(test_rmse.round(3))+', test mape: '+str(test_mape.round(3))+
                               '\nw: '+str(w_size)+' ,p: '+str(p_size)+', epoch: '+str(e)+', tr: '+str(test_rate) + ' mae')
                        plt.show()
                        
                        errors.append([e,np.mean(train_rmse),np.mean(train_mape), test_rmse.round(3),test_mape.round(3)])
                    
                if final_test_rmse > test_rmse:
                    final_test_rmse = test_rmse
                    final_test_mape = test_mape
            m.train()
        
        error['GRU_'+str(i)+'_'+str(z)] = pd.DataFrame(errors, columns = ['epoch', 'train_rmse', 'train_mape', 'test_rmse', 'test_mape'])    

#%%  

# 학습 후 예측 
test_set = pd.read_csv('C:/Users/hyunj/.spyder-py3/Dacon/dataset/dataset/prepossed_test_data/minmax_test_set.csv')
test_data = test_set.values
test_data = torch.FloatTensor(test_data) # test_data.shape = [180, number of features] 
test_data = torch.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))
predict = m(test_data.to(device))
predict = predict.detach().cpu().numpy()
for j in range(p_size):
        predict[:,[j]] = scaler.inverse_transform(predict[:,[j]])
#%%
# Submit
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['Weekly_Sales'] = predict
sample_submission.to_csv('submission.csv',index = False)