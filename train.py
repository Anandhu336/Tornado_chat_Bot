import numpy as np  # Import numpy for numerical operations
import json  # Import json for reading JSON data

import torch  # Import PyTorch for building and training neural networks
import torch.nn as nn  # Import torch.nn to define neural network layers and loss functions
from torch.utils.data import Dataset, DataLoader  # Import Dataset and DataLoader for handling data in batches

from nltk_utils import bag_of_words, tokenize, stem  # Import custom utilities for text processing
from model import NeuralNet  # Import the custom neural network model

# Load the intents file (JSON) containing patterns and corresponding tags (intents)
with open('intents.json', 'r') as f:
    intents = json.load(f)  # Load the JSON data into a Python dictionary

all_words = []  # Initialize an empty list to hold all words in the dataset
tags = []  # Initialize an empty list to hold all unique tags
xy = []  # Initialize an empty list to hold pairs of tokenized patterns and their corresponding tags

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']  # Extract the tag (intent) for the current pattern
    tags.append(tag)  # Add the tag to the tags list
    for pattern in intent['patterns']:
        w = tokenize(pattern)  # Tokenize each word in the sentence
        all_words.extend(w)  # Add the tokenized words to the all_words list
        xy.append((w, tag))  # Add the (words, tag) pair to the xy list

ignore_words = ['?', '.', '!']  # List of words to ignore during processing
all_words = [stem(w) for w in all_words if w not in ignore_words]  # Stem and lowercase each word, ignoring specified punctuation
all_words = sorted(set(all_words))  # Remove duplicates and sort the list of words
tags = sorted(set(tags))  # Sort the list of tags and remove duplicates

# Print some statistics about the processed data
print(len(xy), "patterns")  # Print the number of patterns
print(len(tags), "tags:", tags)  # Print the number of unique tags and the tags themselves
print(len(all_words), "unique stemmed words:", all_words)  # Print the number of unique stemmed words and the words themselves

X_train = []  # Initialize the training data list (input features)
y_train = []  # Initialize the training labels list (output classes)
for (pattern_sentence, tag) in xy:  # Loop through each pair of tokenized sentence and corresponding tag
    bag = bag_of_words(pattern_sentence, all_words)  # Create a bag of words vector for the sentence
    X_train.append(bag)  # Add the bag of words to the training data
    label = tags.index(tag)  # Convert the tag to a numerical label based on its index in the tags list
    y_train.append(label)  # Add the label to the training labels

X_train = np.array(X_train)  # Convert the training data to a numpy array
y_train = np.array(y_train)  # Convert the training labels to a numpy array

# Hyper-parameters
num_epochs = 1000  # Number of epochs (iterations over the entire dataset)
batch_size = 8  # Size of each batch of data
learning_rate = 0.001  # Learning rate for the optimizer
input_size = len(X_train[0])  # Number of input features (size of the bag of words vector)
hidden_size = 8  # Number of neurons in the hidden layer
output_size = len(tags)  # Number of output classes (unique tags)
print(input_size, output_size)  # Print the input size and output size

# Define a custom dataset class for the chatbot data
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)  # Number of samples in the dataset
        self.x_data = X_train  # Input data (features)
        self.y_data = y_train  # Output data (labels)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]  # Support indexing to get a sample by index

    def __len__(self):
        return self.n_samples  # Return the total number of samples

# Create a dataset and data loader for training
dataset = ChatDataset()  # Instantiate the dataset
train_loader = DataLoader(dataset=dataset,  # Create a data loader to load data in batches
                          batch_size=batch_size,  # Set batch size
                          shuffle=True,  # Shuffle the data at each epoch
                          num_workers=0)  # Number of subprocesses to use for data loading (0 means use the main process)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set the device to GPU if available, otherwise use CPU

model = NeuralNet(input_size, hidden_size, output_size).to(device)  # Instantiate the neural network model and move it to the selected device

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # Define the loss function (CrossEntropyLoss for multi-class classification)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Define the optimizer (Adam) and set the learning rate

# Train the model
for epoch in range(num_epochs):  # Loop over the number of epochs
    for (words, labels) in train_loader:  # Loop over each batch of data
        words = words.to(device)  # Move the input data to the selected device
        labels = labels.to(dtype=torch.long).to(device)  # Move the labels to the selected device and ensure they are of type long

        outputs = model(words)  # Forward pass: compute the model output for the input data
        loss = criterion(outputs, labels)  # Compute the loss between the predicted and actual labels

        optimizer.zero_grad()  # Clear the previous gradients
        loss.backward()  # Backward pass: compute the gradients
        optimizer.step()  # Update the model parameters based on the gradients

    if (epoch + 1) % 100 == 0:  # Every 100 epochs
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')  # Print the current epoch and loss

print(f'final loss: {loss.item():.4f}')  # Print the final loss after training

# Save the trained model and related data
data = {
    "model_state": model.state_dict(),  # Save the model's learned parameters
    "input_size": input_size,  # Save the input size
    "hidden_size": hidden_size,  # Save the hidden layer size
    "output_size": output_size,  # Save the output size
    "all_words": all_words,  # Save the list of all words (vocabulary)
    "tags": tags  # Save the list of all tags (classes)
}

FILE = "data.pth"  # Specify the file name for saving the model
torch.save(data, FILE)  # Save the model and related data to a file

print(f'training complete. file saved to {FILE}')  # Print a message indicating that training is complete and the model has been saved
