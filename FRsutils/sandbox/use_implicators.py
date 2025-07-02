from FRsutils.core.implicators import Implicator
import tests.synthetic_data_store as sds
import numpy as np

data = sds.get_implicator_scalar_testsets()
a_b = data[0]['a_b'].T
vals = data[0]['expected']

# Create an implicator by name (e.g., 'gaines')
imp = Implicator.create("reichenbach")
result = imp(a_b[0], a_b[1])

print("Gaines Implicator Result:", result)

imp = Implicator.create("luk")  # alias for lukasiewicz
print("Łukasiewicz Implicator:", imp(0.3, 0.7))

imp = Implicator.create("gaines")  # alias for lukasiewicz
print("Łukasiewicz Implicator:", imp(0.3, 0.7))

# # Yager implicator with p = 2
# imp = Implicator.create("yager", p=2.0)
# print("Yager Implicator (p=2):", imp(0.5, 0.8))

# # Weber implicator with lambda = 0.7
# imp = Implicator.create("weber", lambd=0.7)
# print("Weber Implicator (λ=0.7):", imp(0.9, 0.2))

# # Frank implicator with s = 2
# imp = Implicator.create("frank", s=2.0)
# print("Frank Implicator (s=2):", imp(0.4, 0.6))

# imp = Implicator.create("yager", p=2.0)
# state = imp.to_dict()
# print("Serialized:", state)

# # Deserialize
# restored = Implicator.from_dict(state)
# print("Restored Result:", restored(0.5, 0.6))

# available = Implicator.list_available()
# for name, aliases in available.items():
#     print(f"{name}: aliases = {aliases}")

# print(Implicator.help("frank"))