"""
@file
@brief Factory function to build similarity functions by name.
"""

from FRsutils.core.similarities import LinearSimilarity, GaussianSimilarity, SimilarityFunction
from FRsutils.utils.validation_utils import validate_similarity_choice

def build_similarity(name: str, gaussian_similarity_sigma) -> SimilarityFunction:
    """
    @brief Instantiates a similarity function object.

    @param name Name of the similarity function ('linear', 'gaussian').
    @param kwargs Optional keyword args (e.g., sigma for GaussianSimilarity)

    @return Instance of SimilarityFunction.

    @throws ValueError If the name or arguments are invalid.
    """
    name = validate_similarity_choice(name)
    
    # Instantiate the similarity object
    if name == 'linear':
        return LinearSimilarity()
    elif name == 'gaussian':
        return GaussianSimilarity(sigma=gaussian_similarity_sigma)
    else:
        raise ValueError(f"Unknown similarity function: {name}")
