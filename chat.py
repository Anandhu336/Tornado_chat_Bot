import random  # Import the random module to randomly select responses
import json  # Import the json module to handle JSON data

import torch  # Import the PyTorch library

from model import NeuralNet  # Import the NeuralNet class from the model module
from nltk_utils import bag_of_words, tokenize  # Import helper functions for tokenization and bag-of-words processing

# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the intents JSON file containing the predefined intents and responses
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load the trained model's data
FILE = "data.pth"
data = torch.load(FILE)

# Extract model parameters and data from the loaded file
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize the model with the loaded parameters and move it to the appropriate device (CPU/GPU)
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)  # Load the trained model state
model.eval()  # Set the model to evaluation mode (disable dropout, etc.)

# Define the bot's name
bot_name = "Bot"

# Define a function to generate a response based on the input message
def get_response(msg):
    sentence = tokenize(msg)  # Tokenize the input sentence into words
    X = bag_of_words(sentence, all_words)  # Convert the sentence into a bag-of-words vector
    X = X.reshape(1, X.shape[0])  # Reshape the vector to match the input format of the model
    X = torch.from_numpy(X).to(device)  # Convert the numpy array to a PyTorch tensor and move it to the appropriate device

    output = model(X)  # Pass the input through the model to get the output
    _, predicted = torch.max(output, dim=1)  # Get the index of the highest scoring class (predicted tag)

    tag = tags[predicted.item()]  # Convert the predicted index to the corresponding tag

    probs = torch.softmax(output, dim=1)  # Apply softmax to get the probabilities of each class
    prob = probs[0][predicted.item()]  # Get the probability of the predicted class
    if prob.item() > 0.75:  # If the probability is greater than 0.75 (confidence threshold)
        for intent in intents['intents']:  # Iterate through the intents in the JSON data
            if tag == intent["tag"]:  # If the predicted tag matches the intent tag
                return random.choice(intent['responses'])  # Return a random response from the matched intent

    return "I do not understand..."  # Return a default response if the confidence is too low
