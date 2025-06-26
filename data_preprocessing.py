import numpy as np
import requests

# Descargar el dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text

# Crear vocabulario
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mapeos
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# Convertir un caracter a un vector one-hot
def char_to_one_hot(char_idx, vocab_size):
    vec = np.zeros((vocab_size, 1))
    vec[char_idx] = 1
    return vec

# Convertir un vector one-hot a un caracter
def one_hot_to_char(one_hot_vec, ix_to_char):
    idx = np.argmax(one_hot_vec)
    return ix_to_char[idx]