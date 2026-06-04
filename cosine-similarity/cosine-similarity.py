import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a=np.array(a)
    b=np.array(b)
    if(np.linalg.norm(a)==0 or np.linalg.norm(b)==0):
        cos=0.0
    else:
        cos=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos
    pass