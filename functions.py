import numpy as np

# --- Activation Functions ---
def tanh(x):
    return np.tanh(x) 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if x.ndim == 1:
        exp_x = np.exp(x - np.max(x))
    else: # Assumes x is 2D -> (batch_size, num_features)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    
    return exp_x / np.sum(exp_x, axis=x.ndim - 1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

# --- Loss Functions ---
def categorical_cross_entropy_loss(y_true, y_pred, epsilon=1e-10):
    # Clip predictions to avoid log(0) and log(~0) issues
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # Compute cross-entropy for each sample and then average
    # The sum is over the classes (axis=1), then mean over the batch (axis=0)
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(loss)

def binary_cross_entropy_loss(y_true, y_pred, epsilon=1e-10):
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

