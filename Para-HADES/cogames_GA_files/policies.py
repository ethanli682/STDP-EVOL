import torch
import torch.nn as nn


class MLP_heb(nn.Module):
    "MLP, no bias"
    def __init__(self, input_space, action_space):
        super(MLP_heb, self).__init__()

        self.fc1 = nn.Linear(input_space, 64, bias=False)
        self.fc2 = nn.Linear(64, 32, bias=False)
        self.fc3 = nn.Linear(32, action_space, bias=False)

    def forward(self, ob):
        state = torch.as_tensor(ob[0]).float().detach()
        
        x1 = torch.tanh(self.fc1(state))   
        x2 = torch.tanh(self.fc2(x1))
        o = self.fc3(x2)  
         
        return state, x1, x2, o
        # return state, self.fc1(state), self.fc2(x1), self.fc3(x2)  
    

class MLP_static(nn.Module):
    """Feed-forward MLP, no bias, no plasticity. Weights are the genes."""
    def __init__(self, input_space, action_space, hidden=(128, 64)):
        super().__init__()
        h1, h2 = hidden
        self.fc1 = nn.Linear(input_space, h1, bias=False)
        self.fc2 = nn.Linear(h1,          h2, bias=False)
        self.fc3 = nn.Linear(h2, action_space, bias=False)

    def forward(self, ob):
        x = torch.as_tensor(ob[0]).float().detach()
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


class GRU_static(nn.Module):
    """Projection -> GRUCell -> linear. No bias. Hidden state carried by caller."""
    def __init__(self, input_space, action_space, proj=64, hidden=32):
        super().__init__()
        self.hidden_size = hidden
        self.proj = nn.Linear(input_space, proj, bias=False)
        self.gru  = nn.GRUCell(proj, hidden, bias=False)
        self.out  = nn.Linear(hidden, action_space, bias=False)

    def init_hidden(self, batch=1):
        return torch.zeros(batch, self.hidden_size)

    def forward(self, ob, h):
        x = torch.as_tensor(ob[0]).float().detach().unsqueeze(0)
        x = torch.tanh(self.proj(x))
        h = self.gru(x, h)
        return self.out(h).squeeze(0), h


class CNN_heb(nn.Module):
    "CNN+MLP with n=input_channels frames as input. Non-activated last layer's output"
    def __init__(self, input_channels, action_space_dim):
        super(CNN_heb, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=3, stride=1, bias=False)   
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5, stride=2, bias=False)
        
        self.linear1 = nn.Linear(648, 128, bias=False) 
        self.linear2 = nn.Linear(128, 64, bias=False)
        self.out = nn.Linear(64, action_space_dim, bias=False)
    
    
    def forward(self, ob):
        
        state = torch.as_tensor(ob.copy())
        state = state.float()
        
        x1 = self.pool(torch.tanh(self.conv1(state)))
        x2 = self.pool(torch.tanh(self.conv2(x1)))
        
        x3 = x2.view(-1)
        
        x4 = torch.tanh(self.linear1(x3))   
        x5 = torch.tanh(self.linear2(x4))
        
        o = self.out(x5)

        return x3, x4, x5, o
        # return self.pool(self.conv2(x1)).view(-1), self.linear1(x3), self.linear2(x4), o
        
