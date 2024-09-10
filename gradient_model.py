from base_model import BaseModel
from torch import optim
f
import numpy as np

class GradientModel(BaseModel):
    def __init__(self, input_size=3, hidden_sizes=[5, 4], output_size=1, 
                 weight_range=[-1, 1], bias_range=[-1, 1]):
        super().__init__(input_size, hidden_sizes, output_size, weight_range, bias_range)

    def train_model(self, X, y, num_epochs=1, learning_rate=1e-4):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_history = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            y_pred = self(X)
            loss = self.get_loss(y_pred, y)
            loss.backward()

            optimizer.step()

            print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
            loss_history.append(loss.item())

        return np.array(loss_history)



