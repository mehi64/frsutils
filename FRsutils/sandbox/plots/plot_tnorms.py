import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import FRsutils.core.tnorms as tn

# --- Configuration ---
n = 50  # Grid resolution
tnorm_type = "min"
tnorm_type = "product"
tnorm_type = "luk"
tnorm_type = "drastic"
tnorm_type = "einstein"
tnorm_type = "hamacher"
tnorm_type = "nilpotent"
tnorm_type = "yager"

p_value = 2.0

# --- Create grid ---
b_vals = np.linspace(0, 1, n)
a_vals = np.linspace(0, 1, n)
AVals, BVals = np.meshgrid(a_vals, b_vals)

# --- Instantiate T-norm ---
tnrm = tn.TNorm.create(tnorm_type, p=p_value)
print("Using TNorm:", tnrm.name)

# --- Compute T-norm values ---
TNVals = tnrm.__call__(AVals, BVals)

# --- Prepare plot ---
fig = plt.figure(figsize=(14, 6))

# --- 3D Scatter Plot ---
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
colors = TNVals.ravel()
scatter = ax3d.scatter(AVals.ravel(), BVals.ravel(), TNVals.ravel(),
                       c=colors, cmap='viridis', marker='.')

# Titles and labels
params = tnrm.describe_params_detailed()
param_str = " ".join(f"{k}={v['value']}" for k, v in params.items())
title = f"{tnrm.name} ({param_str})" if param_str else tnrm.name
ax3d.set_title("3D Scatter: " + title)
ax3d.set_xlabel("a")
ax3d.set_ylabel("b")
ax3d.set_zlabel("T(a, b)")
ax3d.set_zlim(0, 1)

# Colorbar for 3D
mappable = plt.cm.ScalarMappable(cmap='viridis')
mappable.set_array(colors)
fig.colorbar(mappable, ax=ax3d, shrink=0.6, aspect=10, label="T-norm Output")

# --- 2D Contour Plot ---
ax2d = fig.add_subplot(1, 2, 2)
contour = ax2d.contourf(AVals, BVals, TNVals, levels=50, cmap='viridis')
ax2d.set_title("2D Contour: " + title)
ax2d.set_xlabel("a")
ax2d.set_ylabel("b")

# Colorbar for 2D
fig.colorbar(contour, ax=ax2d, shrink=0.9, aspect=10, label="T-norm Output")

plt.tight_layout()
plt.show()
