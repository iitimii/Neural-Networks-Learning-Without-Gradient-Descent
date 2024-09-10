from base_model import BaseModel

import numpy as np

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradientModel(BaseModel):
    def __init__(self, input_size=3, hidden_sizes=[5, 4], output_size=1, 
                 weight_range=[-1, 1], bias_range=[-1, 1]):
        super().__init__(input_size, hidden_sizes, output_size, weight_range, bias_range)
    
    def train_model(self, X, y, num_epochs=1, learning_rate=1e-1, patience=5, factor=0.1):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
        loss_history = []
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = self(X)
            loss = self.get_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
            loss_history.append(loss.item())
    
            scheduler.step(loss)
        
        return np.array(loss_history)