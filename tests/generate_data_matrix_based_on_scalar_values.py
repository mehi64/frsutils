import numpy as np
import FRsutils.core.tnorms as tn


def apply_tnorm_columnwise(A: np.ndarray, 
                           B: np.ndarray, 
                           tnorm_name,
                           **kwargs) -> np.ndarray:
    """
    Applies the specified T-norm column-wise between two nxn matrices A and B.

    For each column i, it:
    - Forms an nx2 matrix by taking A[:, i] and B[:, i]
    - Applies T-norm elementwise on each row
    - Stores result as column i in output matrix

    @param A: First input matrix (nxn)
    @param B: Second input matrix (nxn)
    @param tnorm_name: Alias of the T-norm to apply (e.g., "lukasiewicz", "yager", "lambda")
    @param kwargs: Optional parameters for parameterized T-norms (e.g., p=0.8 for Yager)
    @return: Matrix of shape (n, n) with T-norm applied column-wise
    """
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of same shape (nxn)")

    n = A.shape[0]
    output = np.zeros_like(A)

    tnorm = tn.TNorm.create(tnorm_name, **kwargs)

    for i in range(n):
        a = A[:, i]
        b = B[:, i]
        output[:, i] = tnorm(a, b)

    return output
    
    
    

   
similarity_matrix = np.array([
        [1.0,     0.2673,  0.25456, 0.1197,  0.09504],
        [0.2673,  1.0,     0.0658,  0.1624,  0.054  ],
        [0.25456, 0.0658,  1.0,     0.3157,  0.53217],
        [0.1197,  0.1624,  0.3157,  1.0,     0.53872],
        [0.09504, 0.054,   0.53217, 0.53872, 1.0     ]
    ])

label_mask = np.array([
        [1.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0]
    ])

val = apply_tnorm_columnwise(similarity_matrix,
                        label_mask, 
                        tnorm_name="drastic_product", p= 0.1)
print(val)
