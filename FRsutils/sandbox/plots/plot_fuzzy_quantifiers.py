import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../core')))

import fuzzy_quantifiers as fq

# Define parameters
alpha = 0.3
beta = 0.7

# Input values from 0 to 1
x_vals = np.linspace(0, 1, 200)
y_vals = fq.fuzzy_quantifier_quad(x_vals, alpha, beta)
y_vals = fq.fuzzy_quantifier1(x_vals, alpha, beta)

# Plot
plt.plot(x_vals, y_vals, label=f'Q({alpha}, {beta})(x)', color='blue')
plt.title("Smooth Fuzzy Quantifier")
plt.xlabel("x")
plt.ylabel("Q(x)")
plt.grid(True)
plt.legend()
plt.show()
