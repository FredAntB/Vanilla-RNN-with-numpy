import numpy as np
from data_preprocessing import text, char_to_ix, ix_to_char, vocab_size
from model import VanillaRNN
from functions import categorical_cross_entropy_loss
from functions import softmax 

# Configuration for testing
hidden_size = 100
seq_length = 300

# Create and load the model
model = VanillaRNN(hidden_size, vocab_size)
model.load_model("rnn_trained.npz")

# Choose a sequence from the text to test
start_idx = len(text) - seq_length - 1
inputs = [char_to_ix[ch] for ch in text[start_idx:start_idx + seq_length]]
targets = [char_to_ix[ch] for ch in text[start_idx + 1:start_idx + seq_length + 1]]

# Initial hidden state
hprev = np.zeros((hidden_size, 1))

# --- Forward Pass ---
loss, xs, hs, ps = model.forward(inputs, targets, hprev)
print(f"\nTest Loss on sequence of {seq_length} chars: {loss:.4f}")

# --- Generate text from the model ---
sample_length = 300
sample_ix = model.sample(hprev, inputs[0], sample_length)
generated_text = ''.join(ix_to_char[ix] for ix in sample_ix)

print("\nGenerated Text Sample:\n")
print(generated_text)
