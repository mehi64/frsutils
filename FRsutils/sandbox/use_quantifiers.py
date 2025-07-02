from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier as fq
import numpy as np

linear_q1 = fq.create("linear", alpha=0.5, beta=1.6, validate_inputs=True)
quad_q1 = fq.create("quadratic", alpha=0.1, beta=0.6, validate_inputs=False) 

x_vals = np.linspace(0, 1, 200)

y_vals_linear = linear_q1(x_vals)
y_vals_quadratic = quad_q1(x_vals)

print(1)