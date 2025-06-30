import numpy as np

# Debug function to check model parameters
def debug_forward(model, text, char_to_ix, ix_to_char, seq_length=10):
    print("\n--- Debugging forward pass ---")

    # Toma una secuencia corta del texto
    inputs = [char_to_ix[ch] for ch in text[:seq_length]]
    targets = [char_to_ix[ch] for ch in text[1:seq_length+1]]

    hprev = np.zeros((model.hidden_size, 1))

    loss, xs, hs, ps = model.forward(inputs, targets, hprev)

    for t in range(seq_length):
        print(f"\nStep {t}:")
        print(f"  Input char: '{ix_to_char[inputs[t]]}' (index {inputs[t]})")
        print(f"  Target char: '{ix_to_char[targets[t]]}' (index {targets[t]})")

        print(f"  Input one-hot shape: {xs[t].shape}, sum: {np.sum(xs[t])}")
        print(f"  Output probs sum: {np.sum(ps[t])}")
        print(f"  Output probs (top 5):")
        top5_idx = np.argsort(ps[t].ravel())[::-1][:5]
        for idx in top5_idx:
            print(f"    '{ix_to_char[idx]}': {ps[t][idx][0]:.4f}")
    
    print(f"\nTotal loss over sequence: {loss:.4f}")

def debug_backward(model, text, char_to_ix, ix_to_char, seq_length=10):
    print("\n--- Debugging backward pass ---")

    # Toma una secuencia corta
    inputs = [char_to_ix[ch] for ch in text[:seq_length]]
    targets = [char_to_ix[ch] for ch in text[1:seq_length+1]]
    hprev = np.zeros((model.hidden_size, 1))

    # Forward pass
    loss, xs, hs, ps = model.forward(inputs, targets, hprev)

    # Visualizar los primeros dy
    print("\nGradientes de salida (dy) para los primeros 3 pasos:")
    for t in range(min(3, seq_length)):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1  # derivada de cross-entropy

        print(f"\nPaso {t}:")
        print(f"  Input: '{ix_to_char[inputs[t]]}' -> Target: '{ix_to_char[targets[t]]}' (idx {targets[t]})")
        print(f"  Top 5 predicciones:")
        top5 = np.argsort(ps[t].ravel())[::-1][:5]
        for i in top5:
            print(f"    '{ix_to_char[i]}': {ps[t][i][0]:.4f}")
        print(f"  dy[target]: {dy[targets[t]][0]:.4f} (debería ser negativa)")

        print(f"  Suma abs(dy): {np.sum(np.abs(dy)):.4f}")
        print(f"  Suma dy (esperada≈0): {np.sum(dy):.4f}")

    # Backward pass
    dWxh, dWhh, dWhy, dbh, dby = model.backward(xs, hs, ps, targets)

    # Mostrar resumen de gradientes
    print("\nResumen de gradientes por matriz:")
    def print_stats(name, grad):
        print(f"  {name}: mean={np.mean(grad):.6f}, max={np.max(np.abs(grad)):.6f}")

    print_stats("dWxh", dWxh)
    print_stats("dWhh", dWhh)
    print_stats("dWhy", dWhy)
    print_stats("dbh", dbh)
    print_stats("dby", dby)

    # ¿Se colapsaron los gradientes?
    if all(np.allclose(grad, 0) for grad in [dWxh, dWhh, dWhy, dbh, dby]):
        print("\n Todos los gradientes están en cero. Puede haber un bug en backward.")
    else:
        print("\n Se generaron gradientes distintos de cero.")