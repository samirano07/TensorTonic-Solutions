import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    N,D=X.shape
    w = np.zeros(D)    
    b=0.0
    y=y.reshape(-1)
    for i in range(steps):
        z=np.dot(X,w)+b
        p=_sigmoid(z)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        loss = - (1 / N) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        error=p-y
        dw = (1 / N) * np.dot(X.T, error)
        db = (1 / N) * np.sum(error)
        w-=(lr*dw)
        b-=(lr*db)
    return w,b
pass