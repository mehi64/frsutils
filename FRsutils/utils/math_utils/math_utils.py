import numpy as np
import heapq


def _weighted_random_choice(items_with_weights, random_state):
    """Selects one item based on weights using roulette wheel."""
    if not items_with_weights:
        return None, -1

    items, weights = zip(*items_with_weights)
    weights = np.array(weights)

    # Handle non-positive weights for robustness
    valid_mask = weights > 0
    if not np.any(valid_mask):
        # Fallback: uniform random choice if no positive weights
        idx = random_state.randint(len(items))
        return items[idx], idx

    valid_weights = weights[valid_mask]
    valid_items = [item for item, keep in zip(items, valid_mask) if keep]
    original_indices = [i for i, keep in enumerate(valid_mask) if keep] # Store original index if needed

    total_weight = np.sum(valid_weights)
    # No need to check total_weight <= 0 now, as we ensured positive weights exist

    random_val = random_state.uniform(0, total_weight)
    cumulative_weight = np.cumsum(valid_weights)
    chosen_idx_in_valid = np.searchsorted(cumulative_weight, random_val, side='left')

    # Handle potential index out of bounds if random_val equals total_weight exactly
    chosen_idx_in_valid = min(chosen_idx_in_valid, len(valid_items) - 1)

    return valid_items[chosen_idx_in_valid], original_indices[chosen_idx_in_valid] # Return item and its original index


def _weighted_sampling_without_replacement(population_with_weights, k, random_state):
    """
    Selects k items without replacement based on weights.
    Uses Algorithm A-Res (Reservoir sampling with weighted items) - More efficient.
    Reference: Efraimidis, P. S., & Spirakis, P. G. (2006). Weighted random sampling with a reservoir.
    """
    if k <= 0:
        return [], []
    if not population_with_weights:
        return [], []

    population_size = len(population_with_weights)
    k = min(k, population_size)

    # Calculate key for each item: key = random_uniform^(1/weight)
    keys = []
    original_indices = []
    items = []
    epsilon = 1e-9 # Prevent division by zero or issues with weight=0

    for i, (item, weight) in enumerate(population_with_weights):
        w = max(weight, epsilon) # Ensure weight is positive
        r = random_state.uniform(0.0, 1.0)
        key = r**(1.0 / w)
        keys.append(key)
        original_indices.append(i)
        items.append(item)

    # Use a min-heap to keep track of the k items with the largest keys
    # Store (key, original_index, item)
    reservoir = []
    for i in range(population_size):
        key = keys[i]
        item = items[i]
        original_idx = original_indices[i]

        if len(reservoir) < k:
            heapq.heappush(reservoir, (key, original_idx, item))
        else:
            # If current key is larger than the smallest key in the heap, replace it
            if key > reservoir[0][0]:
                heapq.heapreplace(reservoir, (key, original_idx, item))

    # Extract selected items and indices from the reservoir heap
    selected_items = [item for _, _, item in reservoir]
    selected_indices = [idx for _, idx, _ in reservoir]

    return selected_items, selected_indices
