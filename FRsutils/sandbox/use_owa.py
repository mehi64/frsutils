from FRsutils.core.owa_weights import OWAWeights
import numpy as np

np.set_printoptions(precision=8, suppress=True)

# Linear weights (default)
owa = OWAWeights.create("linear")
# owa = OWAWeights.create("exp", base=3.0)
# owa = OWAWeightStrategy.create("harmonic")
# owa = OWAWeightStrategy.create("log")

print(owa.weights(3, 'desc'))

# print(owa.lower_weights(5))
# print(owa.lower_weights(8))
# print(owa.lower_weights(10))
# print(owa.lower_weights(13))

# print(np.sum(owa.lower_weights(5)))
# print(np.sum(owa.lower_weights(8)))
# print(np.sum(owa.lower_weights(10)))
# print(np.sum(owa.lower_weights(13)))



# # exponential weights
# owa = OWAWeightStrategy.create("exp")
# weights = owa.weights(n=5, descending=False)  # same as upper_weights
# print(weights)

# print(np.sum(weights))

# # Harmonic weights
# owa = OWAWeightStrategy.create("harmonic")
# weights = owa.weights(n=5, descending=False)  # same as upper_weights
# print(weights)

# # logarithmic weights
# owa = OWAWeightStrategy.create("log")
# weights = owa.weights(n=5, descending=False)  # same as upper_weights
# print(weights)