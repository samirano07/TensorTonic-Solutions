import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x=np.array(x)
    if(x.ndim==2):
        x_max=np.max(x,axis=1,keepdims=True)
        total=np.sum(np.exp(x-x_max),axis=1,keepdims=True)
        softmax=np.exp(x-x_max)/total
    else:
        x_max=np.max(x)
        total=np.sum(np.exp(x-x_max))
        softmax=np.exp(x-x_max)/total
    return softmax
    pass
