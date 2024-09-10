from base_model import BaseModel
import torch
import numpy as np

class SearchModel(BaseModel):
    def __init__(self, input_size=3, hidden_sizes=[5, 4], output_size=1, 
                 weight_range=[-1, 1], bias_range=[-1, 1],
                 weight_range_size=5, bias_range_size=5):
        
        super().__init__(input_size, hidden_sizes, output_size, weight_range, bias_range)
        self.weight_range_size = weight_range_size
        self.bias_range_size = bias_range_size

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

    
    def _search_single_parameter(self, X, y, param_name, sample_range, *indices):
        best_loss = float('inf')
        best_value = None
        for value in sample_range:
            self.set_single_parameter(param_name, *indices, value=value)

            y_pred = self(X)
            loss = self.get_loss(y_pred, y).item()

            if loss < best_loss and not np.isnan(loss):
                best_loss = loss
                best_value = value

        if best_value is not None:
            self.set_single_parameter(param_name, *indices, value=best_value)


    def search_parameter(self, X, y, param_name):
        param = self.state_dict()[param_name]
        sample_range = self.weight_range if 'weight' in param_name else self.bias_range
        if len(param.shape) == 2:  # weight
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    self._search_single_parameter(X, y, param_name, sample_range, i, j)
        elif len(param.shape) == 1:  # bias
            for i in range(param.shape[0]):
                self._search_single_parameter(X, y, param_name, sample_range, i)


    def search(self, X, y, max_iterations=1, early_stopping=True):
        with torch.no_grad():
            history = []
            for iteration in range(max_iterations):
                y_pred = self(X)
                initial_loss = self.get_loss(y_pred, y).item()

                for name, param in self.named_parameters():
                    self.search_parameter(X, y, name)

                y_pred = self(X)
                final_loss = self.get_loss(y_pred, y).item()

                print(f"Iteration {iteration + 1}: Loss improved from {initial_loss:.4f} to {final_loss:.4f}")
                if np.isclose(initial_loss, final_loss) and early_stopping:
                    print("Search converged. Stopping early.")
                    break
                history.append(final_loss)
            return np.array(history)
