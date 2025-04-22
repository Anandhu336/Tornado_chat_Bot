import nltk  # Import the Natural Language Toolkit (nltk) library for text processing
import ssl  # Import the ssl module to handle secure connections

# Try to set up an unverified HTTPS context to bypass SSL certificate verification
try:
    _create_unverified_https_context = ssl._create_unverified_context  # Attempt to create an unverified SSL context
except AttributeError:  # If the system does not support creating an unverified context
    pass  # Do nothing and proceed
else:
    ssl._create_default_https_context = _create_unverified_https_context  # Set the unverified context as the default HTTPS context

# Download nltk resources via the unverified SSL context (useful in environments with SSL certificate issues)
nltk.download()  # Opens the NLTK downloader GUI, allowing you to download NLTK datasets and models
