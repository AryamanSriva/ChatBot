# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import os

# Set TensorFlow to suppress GPU errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_model(input_shape, output_shape):
    """Create and compile the neural network model"""
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))

    # Compile the model
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=5):
    """Train the model and return the history"""
    hist = model.fit(np.array(X_train),
                     np.array(y_train),
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_data=(X_val, y_val),
                     verbose=1)
    
    # Save the trained model
    model.save('trained_model.h5')
    
    return hist

def plot_training_curves(history):
    """Plot the training and validation loss/accuracy curves"""
    plt.rcParams["figure.figsize"] = (12, 8)
    N = np.arange(0, len(history.history["loss"]))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.plot(N, history.history['accuracy'], label="accuracy")
    plt.plot(N, history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig('training_curves.png')
    plt.show()
