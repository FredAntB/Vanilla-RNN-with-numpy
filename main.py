from data_preprocessing import text, char_to_ix, ix_to_char, vocab_size
from train import train_rnn

if __name__ == "__main__":
    trained_model = train_rnn(
        text=text,
        char_to_ix=char_to_ix,
        ix_to_char=ix_to_char,
        vocab_size=vocab_size,
        hidden_size=256,
        seq_length=200,
        learning_rate=1e-4,
        num_epochs=20,
        debug=False,
        save_path="rnn_trained.npz"
    )
