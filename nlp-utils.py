# nlp_utils.py
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import json
import numpy as np

def load_data(filename):
    """Load and parse the intents JSON file"""
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    with open(filename) as data:
        return json.loads(data.read())

def preprocess_data(intents):
    """Tokenize and prepare training data"""
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!']
    lemmatizer = WordNetLemmatizer()

    # Process each pattern in the intents
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize and clean the words
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    # Save the processed data
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))
    
    return words, classes, documents

def create_training_data(words, classes, documents):
    """Create training data for the neural network"""
    lemmatizer = WordNetLemmatizer()
    training = []
    output_empty = [0] * len(classes)

    # Create the bag of words for each pattern
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        
        for w in words:
            if w in pattern_words:
                bag.append(1)
            else:
                bag.append(0)
        
        # Output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    # Shuffle and split the data for training and validation
    import random
    random.shuffle(training)
    training = np.array(training, dtype='object')
    
    # Split training and validation sets (70% train, 30% validation)
    split_point = int(0.7 * len(training))
    X_train = list(training[:split_point, 0])
    y_train = list(training[:split_point, 1])
    X_val = list(training[split_point:, 0])
    y_val = list(training[split_point:, 1])

    return X_train, y_train, X_val, y_val
