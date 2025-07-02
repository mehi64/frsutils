import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import FRsutils.core.implicators as impl

# --- Configuration ---
n = 200  # Grid resolution
# implicator_type = "goguen"
implicator_type = "luk"
# implicator_type = "goedel"
# implicator_type = "kd"
# implicator_type = "reichenbach"
# implicator_type = "rescher"
# implicator_type = "yager"
# implicator_type = "weber"
# implicator_type = "fodor"

# --- Create grid over [0,1] × [0,1] ---
a_vals = np.linspace(0, 1, n)
b_vals = np.linspace(0, 1, n)
A_vals, B_vals = np.meshgrid(a_vals, b_vals)

# --- Create implicator object ---
implicator = impl.Implicator.create(implicator_type)
print("Using Implicator:", implicator.name)

# --- Compute implication values I(a, b) ---
I_vals = implicator(A_vals, B_vals)

# --- Prepare figure with 3D and 2D views ---
fig = plt.figure(figsize=(14, 6))

# --- 3D Surface Scatter Plot ---
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
colors = I_vals.ravel()
scatter = ax3d.scatter(A_vals.ravel(), B_vals.ravel(), I_vals.ravel(),
                       c=colors, cmap='plasma', marker='.')

# Title and labels
params = implicator.describe_params_detailed()
param_str = " ".join(f"{k}={v['value']}" for k, v in params.items())
title = f"{implicator.name} ({param_str})" if param_str else implicator.name
ax3d.set_title("3D Scatter: " + title)
ax3d.set_xlabel("a (antecedent)")
ax3d.set_ylabel("b (consequent)")
ax3d.set_zlabel("I(a, b)")
ax3d.set_zlim(0, 1)

# Colorbar for 3D
mappable = plt.cm.ScalarMappable(cmap='plasma')
mappable.set_array(colors)
fig.colorbar(mappable, ax=ax3d, shrink=0.6, aspect=10, label="Implicator Output")

# --- 2D Contour Plot ---
ax2d = fig.add_subplot(1, 2, 2)
contour = ax2d.contourf(A_vals, B_vals, I_vals, levels=50, cmap='plasma')
ax2d.set_title("2D Contour: " + title)
ax2d.set_xlabel("a (antecedent)")
ax2d.set_ylabel("b (consequent)")

# Colorbar for 2D
fig.colorbar(contour, ax=ax2d, shrink=0.9, aspect=10, label="Implicator Output")

plt.tight_layout()
plt.show()
