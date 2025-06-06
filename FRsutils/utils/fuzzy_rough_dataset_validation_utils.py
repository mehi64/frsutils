import numpy as np


def compatible_dataset_with_FuzzyRough(X: np.ndarray, y: np.ndarray) -> None:
        """
        @brief Validates a dataset consisting of a 2D array `X` and a 1D array `y`
        to be sure it is a valid dataset for fuzzy-rough calculations.

        @details This function ensures:
        - `X` is a 2D NumPy array of floating-point numbers.
        - All elements in `X` are within the range [0.0, 1.0].
        - `y` is a 1D NumPy array.
        - The length of `y` matches the first dimension of `X`.

        @param X A 2D NumPy array of float values representing features. Must be in range [0.0, 1.0].
        @param y A 1D NumPy array whose length is equal to the number of rowa in `X`.

        @throws TypeError If `X` or `y` is not a NumPy array or if `X` is not float.
        @throws ValueError If the shape, dimensionality, or value range conditions are not satisfied.
        """

        ## Check if X is a NumPy ndarray
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy ndarray.")

        ## Check if X is 2-dimensional
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        ## Check if X has float-type elements
        if not np.issubdtype(X.dtype, np.floating):
            raise TypeError("X elements must be of float type.")

        ## Check if all elements in X are within [0.0, 1.0]
        if np.any(X < 0.0) or np.any(X > 1.0):
            raise ValueError("All elements in X must be in the range [0.0, 1.0].")

        ## Check if y is a NumPy ndarray
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy ndarray.")

        ## Check if y is 1-dimensional
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        ## Check if length of y matches the second dimension of X
        if len(y) != X.shape[0]:
            raise ValueError("Length of y must be equal to the first dimension of X.")

        ## All checks passed
        # print("Dataset is valid.")
