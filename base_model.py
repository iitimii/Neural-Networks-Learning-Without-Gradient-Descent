import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
torch.set_grad_enabled(False)

class MyModel(nn.Module):
    def __init__(self, input_size=3, hidden_sizes=[5, 4], output_size=1, 
                 weight_range=[-1, 1], bias_range=[-1, 1], 
                 weight_range_size=5, bias_range_size=5,
                 num_examples=10):
        super(MyModel, self).__init__()
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
        self.weight_sample_range = np.linspace(weight_range[0], weight_range[1], weight_range_size)
        self.bias_sample_range = np.linspace(bias_range[0], bias_range[1], bias_range_size)

        self.initialize_parameters()

        # Generate sample data
        self.X = torch.randn(num_examples, input_size)
        self.y = self.X.sum(dim=1, keepdim=True)

    def initialize_parameters(self):
        for layer in self.layers:
            nn.init.uniform_(layer.weight, self.weight_range[0], self.weight_range[1])
            nn.init.uniform_(layer.bias, self.bias_range[0], self.bias_range[1])
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
    
    def set_single_parameter(self, param_name, *indices, value):
        if param_name in self.state_dict():
            param = self.state_dict()[param_name]
            if len(indices) == len(param.shape):
                with torch.no_grad():
                    param[indices] = value
            else:
                print(f"Incorrect number of indices for parameter {param_name}.")
        else:
            print(f"Parameter {param_name} not found.")

    def get_loss(self):
        y_pred = self(self.X)
        return F.mse_loss(y_pred, self.y)

    def search_parameter(self, param_name):
        param = self.state_dict()[param_name]
        sample_range = self.weight_sample_range if 'weight' in param_name else self.bias_sample_range
        if len(param.shape) == 2:  # weight
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    self._search_single_parameter(param_name, sample_range, i, j)
        elif len(param.shape) == 1:  # bias
            for i in range(param.shape[0]):
                self._search_single_parameter(param_name, sample_range, i)

    def _search_single_parameter(self, param_name, sample_range, *indices):
        best_loss = float('inf')
        best_value = None
        for value in sample_range:
            self.set_single_parameter(param_name, *indices, value=value)
            loss = self.get_loss().item()
            if loss < best_loss and not np.isnan(loss):
                best_loss = loss
                best_value = value
        if best_value is not None:
            self.set_single_parameter(param_name, *indices, value=best_value)

    def search(self, max_iterations=5, early_stopping=True):
        torch.set_grad_enabled(False)
        history = []
        for iteration in range(max_iterations):
            initial_loss = self.get_loss().item()
            for name, param in self.named_parameters():
                self.search_parameter(name)
            final_loss = self.get_loss().item()
            print(f"Iteration {iteration + 1}: Loss improved from {initial_loss:.4f} to {final_loss:.4f}")
            if np.isclose(initial_loss, final_loss) and early_stopping:
                print("Search converged. Stopping early.")
                break
            history.append(final_loss)
        return np.array(history)
    
    def train_with_grad(self, num_epochs=100, learning_rate=0.001):
        torch.set_grad_enabled(True)
        self.train() 
        history = []
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = self(self.X)
            loss = F.mse_loss(y_pred, self.y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
            history.append(loss.item())
        
        self.eval()  # Set the model back to evaluation mode
        torch.set_grad_enabled(False)
        
        # Final evaluation
        with torch.no_grad():
            final_loss = F.mse_loss(self(self.X), self.y)
            print(f"Final Loss after gradient-based training: {final_loss.item():.4f}")

        return np.array(history)


    def evaluate(self):
        with torch.no_grad():
            predictions = self(self.X)
            loss = F.mse_loss(predictions, self.y)
            print("Final Predictions:", predictions)
            print("Actual values:", self.y)
            print("Final loss:", loss.item())
            
