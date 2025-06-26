import numpy as np

def backward(xs, hs, ps, targets):
    """
    Args:
        xs (list): Input data.
        hs (list): Hidden layer activations.
        ps (list): Output layer predictions.
        targets (list): Target values for the output layer.
    """
    global Wxh, Whh, Why, bh, by
    
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)
    
    dhnext = np.zeros_like(hs[0])
    
    for t in reversed(range(len(xs))):
        # Output layer error
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(hs[t].T, dy)
        dby += dy
        
        # Hidden layer error
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dWxh += np.dot(xs[t].T, dhraw)
        dWhh += np.dot(hs[t-1].T, dhraw)
        dhnext = np.dot(Whh.T, dhraw)
        
    # Clip gradients to avoid exploding gradients
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)
        
    return dWxh, dWhh, dWhy, dbh, dby