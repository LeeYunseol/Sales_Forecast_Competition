import torch 
import torch.nn as nn
from torch.autograd import Variable
from statsmodels.tsa.seasonal import STL
import pandas as pd
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_dim, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        self.layer_1 = nn.Linear(hidden_size, 256)
        self.layer_2 = nn.Linear(256,256)
        self.layer_3 = nn.Linear(256,128)
        self.layer_out = nn.Linear(128, out_dim)
        self.relu = nn.ReLU() #Activation Func
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #은닉상태 0 초기화
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #셀 상태 0 초기화 
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        
        hn = hn.view(-1, self.hidden_size) # Reshaping the data for starting LSTM network
        out = self.relu(hn) #pre-processing for first layer
        out = self.layer_1(out) # first layer
        out = self.relu(out) # activation func relu
        
        out = self.layer_2(out)
        out = self.relu(out)
        
        out = self.layer_3(out)
        out = self.relu(out)
        
        out = self.layer_out(out) #Output layer
        return out
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, out_dim, num_layers, seq_length):
        super(GRU, self).__init__()
        #self.num_classes = num_classes 
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.seq_length = seq_length
        
        self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size*seq_length, out_dim)
        #self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ELU()
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #은닉상태 0 초기화
        
        
        output, hn = self.gru(x, h_0)
        
        
        output = output.reshape(-1, self.hidden_size*self.seq_length).to(device)
        
        
        #out = self.relu(output)
        out = self.fc_1(output)
        out = self.relu(out)
        #out = self.fc(out)
        #out = self.relu(out)

        return out
    
class RNN(nn.Module):
    def __init__(self,  input_size, hidden_size, out_dim, num_layers, seq_length):
        super(RNN, self).__init__()
        #self.num_classes = num_classes 
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.seq_length = seq_length
        
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size*seq_length, out_dim)
        #self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ELU()
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #은닉상태 0 초기화
        
        
        output, hn = self.rnn(x, h_0)
        
        
        output = output.reshape(-1, self.hidden_size*self.seq_length).to(device)
        
        #out= self.relu(output)
        #out = self.relu(output)
        out = self.fc_1(output)
        out = self.relu(out)
        #out = self.fc(out)
        #out = self.relu(out)

        return out
    