import numpy as np
from functions import tanh, softmax, categorical_cross_entropy_loss
from data_preprocessing import vocab_size, char_to_one_hot

class VanillaRNN:
    def __init__(self, hidden_size, vocab_size):
        # Hyperparameters
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Initialize weights with small random values to break symmetry
        # input to hidden
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01 
        # hidden to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01 
        # hidden to output
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01 

        # Initialize biases to zeros
        self.bh = np.zeros((hidden_size, 1)) # hidden bias
        self.by = np.zeros((vocab_size, 1))  # output bias

    def forward(self, inputs, targets, hprev):
        # Dictionaries to store activations/inputs for backpropagation
        xs, hs, ps = {}, {}, {}
        # Store the initial hidden state for backprop
        hs[-1] = np.copy(hprev) 

        loss = 0

        for t in range(len(inputs)):
            # 1. Input layer (one-hot encoding)
            xs[t] = char_to_one_hot(inputs[t], self.vocab_size)

            # 2. Hidden layer calculation
            # h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)
            # wpb -> weight + bias -> alludes the values inside tanh
            wpb = np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh
            hs[t] = tanh(wpb)

            # 3. Output layer calculation
            # y_t = Why @ h_t + by
            # These are the logits
            y_t_raw = np.dot(self.Why, hs[t]) + self.by 
            ps[t] = softmax(y_t_raw) # get probabilities

            # 4. Calculate Loss
            # Get the one-hot target for loss calculation
            target_one_hot = char_to_one_hot(targets[t], self.vocab_size)

            # categorical_cross_entropy_loss expects (batch_size, num_classes)
            # current vectors are (num_classes, 1)
            # so transpose them to (1, num_classes)
            loss += categorical_cross_entropy_loss(target_one_hot.T, ps[t].T)

        return loss, xs, hs, ps

    def get_params(self):
        # Returns a tuple of current model parameters.
        return self.Wxh, self.Whh, self.Why, self.bh, self.by

    def update_params(self, dWxh, dWhh, dWhy, dbh, dby, learning_rate=1e-1):
        # Updates parameters using simple SGD.
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

