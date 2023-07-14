import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Reshape input to (batch_size * sequence_length, input_dim)
        self.flatten = nn.Flatten()
        
        # Building your LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Reshape input to (batch_size * sequence_length, input_dim)
        x = self.flatten(x)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x.unsqueeze(1))
        
        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        
        return out
