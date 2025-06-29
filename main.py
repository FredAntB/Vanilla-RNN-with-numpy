import numpy as np

from data_preprocessing import text, char_to_ix, ix_to_char, vocab_size, char_to_one_hot
from model import VanillaRNN
from optimizers import Adam 

# --- Hyperparameters for Training ---
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25   # sequence length for each training example
learning_rate = 1e-3 # Initial learning rate

# --- Instantiate the Model ---
model = VanillaRNN(hidden_size, vocab_size)

# --- Choose and Instantiate your Optimizer ---
optimizer = Adam(learning_rate=1e-3) 

# --- Training Loop Setup ---
n, p = 0, 0 # n: iteration counter, p: pointer to position in the text data
# Initialize smooth_loss: This is an exponentially weighted moving average of the loss.
# It helps to visualize training progress without being too noisy.
# Initial value is set to the loss if the model predicted randomly (uniform probability).
smooth_loss = -np.log(1.0/vocab_size) * seq_length # Loss at iteration 0 (random guesses)

# --- Main Training Loop ---
while True:
    # Prepare inputs and targets for the current sequence
    # If the data pointer 'p' goes beyond the text length (or it's the very first iteration)
    if p + seq_length + 1 >= len(text) or n == 0: 
        hprev = np.zeros((hidden_size, 1)) # Reset RNN memory (hidden state)
        p = 0 # Go from start of data (epoch restart)

    # Extract current input and target sequence indices
    inputs = [char_to_ix[ch] for ch in text[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in text[p+1:p+seq_length+1]]

    # --- Forward Pass ---
    # Perform the forward pass to get loss and intermediate activations
    loss, xs, hs, ps = model.forward(inputs, targets, hprev)
    
    # Update the smoothed loss for display
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    
    # --- Backward Pass ---
    # Compute gradients with respect to model parameters
    dWxh, dWhh, dWhy, dbh, dby = model.backward(xs, hs, ps, targets)
    
    # --- Optimizer Step ---
    # Collect parameters and gradients into lists/tuples for the optimizer
    params = [model.Wxh, model.Whh, model.Why, model.bh, model.by]
    grads = [dWxh, dWhh, dWhy, dbh, dby]

    # Update parameters using the chosen optimizer
    optimizer.update(params, grads, learning_rate)

    # --- Update Hidden State for Next Iteration ---
    # The last hidden state of the current sequence becomes the initial hidden state
    # for the next sequence. This allows the RNN to carry memory across sequences.
    hprev = hs[len(inputs) - 1]

    # --- Print Progress and Sample Text ---
    if n % 100 == 0: # Print every 100 iterations
        print(f'iter {n}, loss: {smooth_loss:.4f}') # print training loss
        
        # Sample 200 characters from the model to see its current generative ability
        # We use the current hprev and the first character of the current input sequence as seed.
        sample_ix = model.sample(hprev, inputs[0], 200) 
        
        # Convert sampled indices back to characters for display
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n%s\n----' % (txt,)) # Print the generated text nicely formatted

    # --- Update Pointers for Next Iteration ---
    p += seq_length # Move data pointer forward by sequence length
    n += 1 # Increment iteration counter
