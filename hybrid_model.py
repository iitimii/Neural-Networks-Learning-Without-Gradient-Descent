
from search_model import SearchModel
from gradient_model import GradientModel

class HybridModel(SearchModel, GradientModel):
    def __init__(self, input_size=3, hidden_sizes=[5, 4], output_size=1, 
                 weight_range=[-1, 1], bias_range=[-1, 1], 
                 weight_range_size=5, bias_range_size=5):
        
        super().__init__(input_size, hidden_sizes, output_size, weight_range, bias_range)
        self.weight_range_size = weight_range_size
        self.bias_range_size = bias_range_size
        

    def search_and_train(self, X, y, num_epochs=1, learning_rate=1e-4, max_search_iterations=3):
        print('Searching for optimal parameters')
        history_search = self.search(X, y, max_iterations=max_search_iterations, early_stopping=True)
        print('Starting gradient based training')
        history_grad = self.train_model(X, y, num_epochs=num_epochs, learning_rate=learning_rate)

        return history_search, history_grad