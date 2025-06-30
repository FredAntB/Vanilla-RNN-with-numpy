import numpy as np
from debug import debug_forward, debug_backward
from data_preprocessing import text, char_to_ix, ix_to_char, vocab_size
from model import VanillaRNN
from optimizers import Adam 

def train_rnn(
    text,
    char_to_ix,
    ix_to_char,
    vocab_size,
    hidden_size=100,
    seq_length=300,
    learning_rate=1e-3,
    num_epochs=5,
    debug=False,
    debug_seq_length=10,
    save_path="rnn_trained.npz"
):
    # Calculate approximate iterations per epoch
    iterations_per_epoch = (len(text) - (seq_length + 1)) // seq_length 
    if iterations_per_epoch <= 0:
        iterations_per_epoch = 1 

    # Instantiate model and optimizer
    model = VanillaRNN(hidden_size, vocab_size)
    optimizer = Adam()

    print(f"Using optimizer: {type(optimizer).__name__}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Total dataset length: {len(text)} characters")
    print(f"Sequence length (steps per iteration): {seq_length}")
    print(f"Approximate iterations per epoch: {iterations_per_epoch}")
    print(f"Total training for {num_epochs} epochs (~{num_epochs * iterations_per_epoch} iterations).\n")

    global_iteration = 0
    smooth_loss = -np.log(1.0/vocab_size) * seq_length

    if debug:
        print("Running forward pass debug...\n")
        debug_forward(model, text, char_to_ix, ix_to_char, seq_length=debug_seq_length)
        print("\nRunning backward pass debug...\n")
        debug_backward(model, text, char_to_ix, ix_to_char, seq_length=debug_seq_length)

    print("Starting training...\n")

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        hprev = np.zeros((hidden_size, 1))
        data_pointer = 0

        while data_pointer + seq_length + 1 < len(text):
            inputs = [char_to_ix[ch] for ch in text[data_pointer:data_pointer+seq_length]]
            targets = [char_to_ix[ch] for ch in text[data_pointer+1:data_pointer+seq_length+1]]

            loss, xs, hs, ps = model.forward(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            dWxh, dWhh, dWhy, dbh, dby = model.backward(xs, hs, ps, targets)
            params = [model.Wxh, model.Whh, model.Why, model.bh, model.by]
            grads = [dWxh, dWhh, dWhy, dbh, dby]
            optimizer.update(params, grads, learning_rate)

            hprev = hs[len(inputs) - 1]

            if global_iteration % 100 == 0:
                print(f'  Iter {global_iteration}, Epoch {epoch+1}, Loss: {smooth_loss:.4f}')
                sample_ix = model.sample(hprev, inputs[0], 200)
                txt = ''.join(ix_to_char[ix] for ix in sample_ix)
                print('  ----\n%s\n  ----\n' % (txt,))

            data_pointer += seq_length
            global_iteration += 1

    print(f"\nTraining finished after {num_epochs} epochs. Total iterations: {global_iteration}")
    model.save_model(save_path)
    return model

# Opcional: si quieres que se pueda ejecutar directamente
if __name__ == "__main__":
    trained_model = train_rnn(text, char_to_ix, ix_to_char, vocab_size)
