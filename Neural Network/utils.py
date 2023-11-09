import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

# Define the path for saving results
abs_path = os.path.abspath(__file__)
results_dir = os.path.join(os.path.dirname(abs_path), 'try')

# Create the 'Results' directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Function to save results to Excel
def save_results_to_excel(results, filename, columns):
    df = pd.DataFrame(results, columns=columns)
    df.to_excel(os.path.join(results_dir, filename), index=False)

# Create a function to generalize the plotting and reduce code repetition
def plot_results(x, y, title, xlabel, ylabel, filename):
    plt.figure()
    plt.plot(x, y, linestyle='-', marker='o', label=title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

# Define the functions you want to model
def function_1(x):
    return np.exp(-x**2)

def function_2(x1, x2):
    return np.sin(2 * np.pi * x1) * np.cos(0.5 * np.pi * x2)


# Define a function to create a simple feedforward neural network
def create_neural_network(num_hidden_nodes, input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_hidden_nodes, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1)  # Output layer
    ])
    return model

# Train and evaluate the neural network
def train_and_evaluate(input_shape, x_train, y_train, x_test, y_test, x_val, y_val, num_hidden_nodes,
                       num_epochs, num_training_samples):
    model = create_neural_network(num_hidden_nodes, input_shape)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_percentage_error'])

    if len(x_train) > num_training_samples:
        x_train_sampled = x_train[:num_training_samples]
        y_train_sampled = y_train[:num_training_samples]
        x_val_sampled = x_val[:num_training_samples]
        y_val_sampled = y_val[:num_training_samples]
    else:
        x_train_sampled = x_train
        y_train_sampled = y_train
        x_val_sampled = x_val
        y_val_sampled = y_val

    history = model.fit(x_train_sampled, y_train_sampled, epochs=num_epochs, validation_data=(x_val_sampled, y_val_sampled), verbose=1)
    test_loss = model.evaluate(x_test, y_test, verbose=0)
    return test_loss, history