from hmac import new
from re import S
import torch 
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size+hidden_size,hidden_size)
        self.i2o = nn.Linear(input_size+hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,x,state =None):
        if state is None:
            state = torch.zeros(x.size(0),self.hidden_size,device=x.device)
        combined =torch.cat((x,state),dim=1)
        state = torch.tanh(self.i2h(combined))
        output = self.i2o(combined)
        output = self.softmax(output)
        return output,state

class RNN(nn.Module):

    def __init__(self,input_size,hidden_size,output_size,num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnnlayer =nn.ModuleList(
            [RNNCell(input_size,hidden_size,hidden_size)]+
            [RNNCell(hidden_size,hidden_size,hidden_size) for _ in range(num_layers-1)]
        )
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,x,state =None):
        if state is None:
            state =torch.zeros(x.size(0), self.hidden_size, device=x.device)
                    
        for i in range(self.num_layers):
            _, new_state = self.rnnlayer[i](x, state)
            x= new_state 
        output = self.fc_out(x)
        output = self.softmax(output)
        return output,new_state
if __name__ == '__main__':
    rnn = RNN(10,20,30)
    x = torch.randn(5,10)
    hid = torch.randn(5,20)
    output,state = rnn(x)
    print(rnn)
    print(output.shape)


        

