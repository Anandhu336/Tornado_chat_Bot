import torch  # Import the main PyTorch library
import torch.nn as nn  # Import PyTorch's neural network module

# Define a neural network class that inherits from nn.Module
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()  # Call the parent class (nn.Module) constructor
        self.l1 = nn.Linear(input_size, hidden_size)  # Define the first fully connected layer (input to hidden layer)
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Define the second fully connected layer (hidden to hidden layer)
        self.l3 = nn.Linear(hidden_size, num_classes)  # Define the third fully connected layer (hidden to output layer)
        self.relu = nn.ReLU()  # Define the ReLU activation function

    def forward(self, x):
        out = self.l1(x)  # Pass the input data through the first layer
        out = self.relu(out)  # Apply ReLU activation function to the output of the first layer
        out = self.l2(out)  # Pass the result through the second layer
        out = self.relu(out)  # Apply ReLU activation function to the output of the second layer
        out = self.l3(out)  # Pass the result through the third layer (output layer)
        # no activation and no softmax at the end
        return out  # Return the final output (raw scores, no softmax applied)
