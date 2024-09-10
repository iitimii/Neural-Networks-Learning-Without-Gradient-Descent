import torch
from torch import nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, input_size=3, hidden_sizes=[5, 4], output_size=1, 
                 weight_range=[-1, 1], bias_range=[-1, 1]):
        super(BaseModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.weight_range = weight_range
        self.bias_range = bias_range

        self.initialize_parameters()

    def initialize_parameters(self):
        for layer in self.layers:
            nn.init.uniform_(layer.weight, self.weight_range[0], self.weight_range[1])
            nn.init.uniform_(layer.bias, self.bias_range[0], self.bias_range[1])
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
    
    def get_loss(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    def evaluate(self, X, y):
        self.eval()
        with torch.no_grad():
            predictions = self(X)
            loss = F.mse_loss(predictions, y)
            print("MSE loss:", loss.item())
            
