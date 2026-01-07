import numpy as np

#for the hidden layers
def ReLU(X):
    return np.clip(X , 0, None) 

#for the output layer
def softMax(logits):
    exponentials = np.exp(logits)
    return exponentials / np.sum(exponentials, axis=1).reshape(-1, 1)

def forward(X):
    pass