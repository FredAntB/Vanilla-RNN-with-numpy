import numpy as np

class Optimizer:
    def update(self, params, grads, learning_rate):
        raise NotImplementedError("Subclasses must implement the update method")

class SGD(Optimizer):
    def update(self, params, grads, learning_rate):
        for param, grad in zip(params, grads):
            param -= learning_rate * grad

class Adagrad(Optimizer):
    def __init__(self):
        self.m = {} # Dictionary to store squared gradients for each parameter

    def update(self, params, grads, learning_rate):
        # We'll use the memory to store per-parameter squared gradients

        if not self.m: 
            for i, param in enumerate(params):
                # Use a unique key for each parameter based on its index
                self.m[f'param_{i}'] = np.zeros_like(param)
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            mem_key = f'param_{i}'
            self.m[mem_key] += grad * grad
            param -= learning_rate * grad / (np.sqrt(self.m[mem_key]) + 1e-8)

class RMSprop(Optimizer):
    def __init__(self, decay_rate=0.99):
        self.decay_rate = decay_rate
        self.cache = {} # Dictionary to store squared gradients (moving average)

    def update(self, params, grads, learning_rate):
        if not self.cache:
            for i, param in enumerate(params):
                self.cache[f'param_{i}'] = np.zeros_like(param)

        for i, (param, grad) in enumerate(zip(params, grads)):
            cache_key = f'param_{i}'
            self.cache[cache_key] = self.decay_rate * self.cache[cache_key] + (1 - self.decay_rate) * (grad ** 2)
            param -= learning_rate * grad / (np.sqrt(self.cache[cache_key]) + 1e-8)

class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {} # mean of gradients
        self.v = {} # variance of gradients
        self.t = 0  # Time step

    def update(self, params, grads, learning_rate):
        self.t += 1 

        if not self.m: # Initialize moments on first call
            for i, param in enumerate(params):
                self.m[f'param_{i}'] = np.zeros_like(param)
                self.v[f'param_{i}'] = np.zeros_like(param)
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            param_key = f'param_{i}'

            # Update biased moments' estimate
            self.m[param_key] = self.beta1 * self.m[param_key] + (1 - self.beta1) * grad
            self.v[param_key] = self.beta2 * self.v[param_key] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected moments' estimate
            m_hat = self.m[param_key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_key] / (1 - self.beta2 ** self.t)

            param -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
