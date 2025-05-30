import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../core')))

import tnorms as tn

# Create a grid of values
n = 200  # Replace with your desired length (must be even)
arr = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

similarity_vals = np.linspace(0, 1, n)
b_vals = arr
AVals, simVals = np.meshgrid(similarity_vals, b_vals)

# Vectorized computation of the implicator
# TNVals = np.minimum(AVals, simVals)
TNVals = np.multiply(AVals, simVals)


# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot surface with colormap
# surf = ax.plot_surface(A, B, Z, cmap='hot', edgecolor='none')
# Plot the points
ax.scatter(AVals, simVals, TNVals, c='blue', marker='.')

# Add color bar
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Labels and title
ax.set_title("product t-norm")
ax.set_ylabel("A(y)")
ax.set_xlabel("similarity (x,y)")
ax.set_zlabel("T(similarity (x,y), A(y))")

plt.tight_layout()
plt.show()
