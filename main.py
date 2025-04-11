# main.py
import os
from nlp_utils import load_data, preprocess_data, create_training_data
from model import create_model, train_model, plot_training_curves

def main():
    # Configure TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    print("Loading and preprocessing data...")
    # Load the data
    intents = load_data('intents.json')
    
    # Preprocess data
    words, classes, documents = preprocess_data(intents)
    
    # Create training data
    X_train, y_train, X_val, y_val = create_training_data(words, classes, documents)
    
    print(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples")
    
    # Create and train the model
    model = create_model(len(X_train[0]), len(y_train[0]))
    print("Model created. Starting training...")
    
    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Plot the training curves
    print("Training completed. Plotting performance curves...")
    plot_training_curves(history)
    
    print("Training completed successfully!")
    print("Model saved as 'trained_model.h5'")
    print("To use the chatbot, run: python run_chatbot.py")

if __name__ == "__main__":
    main()
