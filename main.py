import numpy as np

from data_preprocessing import text, char_to_ix, ix_to_char, vocab_size
from model import VanillaRNN
from optimizers import Adam 

# --- Hyperparameters for Training ---
hidden_size = 100 # Size of the hidden layer of neurons
seq_length = 25   # Length of sequence per training example
learning_rate = 1e-3 # Learning rate 

# --- Epochs Configuration ---
num_epochs = 20

# Calculate approximate iterations per epoch
# We divide by seq_length because each iteration processes one sequence.
iterations_per_epoch = (len(text) - (seq_length + 1)) // seq_length 
if iterations_per_epoch <= 0: # Handle very small texts or large seq_length
    iterations_per_epoch = 1 

# --- Instantiate the Model ---
model = VanillaRNN(hidden_size, vocab_size)

# --- Choose and Instantiate your Optimizer ---
optimizer = Adam()

print(f"Using optimizer: {type(optimizer).__name__}")
print(f"Learning Rate: {learning_rate}")
print(f"Total dataset length: {len(text)} characters")
print(f"Sequence length (steps per iteration): {seq_length}")
print(f"Approximate iterations per epoch: {iterations_per_epoch}")
print(f"Total training for {num_epochs} epochs (~{num_epochs * iterations_per_epoch} iterations).\n")

# --- Training Loop Setup ---
global_iteration = 0 # This will track the total number of parameter updates across all epochs.

# Initialize smooth_loss as an exponentially weighted moving average for stable monitoring.
smooth_loss = -np.log(1.0/vocab_size) * seq_length # Initial loss for random predictions

print("Starting training...\n")

# --- Main Training Loop (Epochs Approach) ---
for epoch in range(num_epochs):
    print(f"--- Epoch {epoch+1}/{num_epochs} ---")
    
    # Reset RNN memory (hidden state) and data pointer at the beginning of each epoch
    hprev = np.zeros((hidden_size, 1))
    data_pointer = 0 # 'p' is now 'data_pointer' for clarity within the epoch loop

    # Loop through the text for the current epoch, processing it in `seq_length` chunks
    # The condition `data_pointer + seq_length + 1 < len(text)` ensures that
    # both `inputs` (up to `p + seq_length`) and `targets` (up to `p + seq_length + 1`)
    # can be fully extracted without going out of bounds.
    while data_pointer + seq_length + 1 < len(text):
        # Extract current input and target sequence indices
        inputs = [char_to_ix[ch] for ch in text[data_pointer:data_pointer+seq_length]]
        targets = [char_to_ix[ch] for ch in text[data_pointer+1:data_pointer+seq_length+1]]

        # --- Forward Pass ---
        loss, xs, hs, ps = model.forward(inputs, targets, hprev)
        
        # Update the smoothed loss for display
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        
        # --- Backward Pass ---
        dWxh, dWhh, dWhy, dbh, dby = model.backward(xs, hs, ps, targets)
        
        # --- Optimizer Step ---
        params = [model.Wxh, model.Whh, model.Why, model.bh, model.by]
        grads = [dWxh, dWhh, dWhy, dbh, dby]
        optimizer.update(params, grads, learning_rate)

        # --- Update Hidden State for Next Iteration ---
        hprev = hs[len(inputs) - 1]

        if global_iteration % 100 == 0: 
            print(f'  Iter {global_iteration}, Epoch {epoch+1}, Loss: {smooth_loss:.4f}')
            
            # Sample text to qualitatively observe model's learning
            sample_ix = model.sample(hprev, inputs[0], 200) 
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('  ----\n%s\n  ----\n' % (txt,))

        # --- Update Pointers for Next Iteration ---
        data_pointer += seq_length # Move data pointer forward by the sequence length
        global_iteration += 1 # Increment the global iteration counter

print(f"\nTraining finished after {num_epochs} epochs. Total iterations: {global_iteration}")
