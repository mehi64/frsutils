import matplotlib.pyplot as plt
import numpy as np

from FRsutils.core.fuzzy_quantifiers import fuzzy_quantifier_linear, fuzzy_quantifier_quadratic

# Define parameters
alpha = 0.2
beta = 0.7

# Input values from 0 to 1
x_vals = np.linspace(0, 1, 200)
y_vals_quadratic = fuzzy_quantifier_quadratic(x_vals, alpha, beta)
y_vals_linear = fuzzy_quantifier_linear(x_vals, alpha, beta)

# # Plot
# plt.plot(x_vals, y_vals_linear, label=f'Q({alpha}, {beta})(x)', color='blue')
# plt.title("Smooth Fuzzy Quantifier")
# plt.xlabel("x")
# plt.ylabel("Q(x)")
# plt.grid(True)
# plt.legend()
# plt.show()

# Create side-by-side plots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=True)

# Plot for quadratic fuzzy quantifier
axes[0].plot(x_vals, y_vals_quadratic, color='blue')
axes[0].set_title("Quadratic Fuzzy Quantifier")
axes[0].set_xlabel("x")
axes[0].set_ylabel("Quantifier Value")
axes[0].grid(True)

# Plot for linear fuzzy quantifier
axes[1].plot(x_vals, y_vals_linear, color='green')
axes[1].set_title("Linear Fuzzy Quantifier")
axes[1].set_xlabel("x")
axes[1].grid(True)

plt.tight_layout()
plt.show()