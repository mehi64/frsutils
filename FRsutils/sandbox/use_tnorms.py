import FRsutils.core.tnorms as tn
import tests.synthetic_data_store as sds
import numpy as np

# data = sds.get_tnorm_scalar_testsets()
# a_b = data[0]['a_b'].T
# vals = data[0]['expected']

# tnorm1 = tn.TNorm.create('einstein')
# tnorm2 = tn.TNorm.create('prod')
# tnorm3 = tn.TNorm.create('yg', strict=False, p=2.3, f=0.8)

# results = tnorm1(a_b[0], a_b[1])
# print(results)

# # params = tnorm3.describe_params_detailed()
# # hlp =tnorm3.help()
# # print(hlp)
# # nme = tnorm3.name

# # dic1 = tnorm3.to_dict()

# # tnorm_instance = tn.TNorm.from_dict(dic1)

# # print(1)


# arr1 = np.array([0.7, 0.4])
# arr2 = np.array([0.9, 0.5])

# arr3 = np.array([[0.7, 0.4, 0.2], [0.1, 0.2, 0.3]])
# arr4 = np.array([[0.9, 0.5, 0.4], [0.6, 0.7, 0.8]])

# # # print(tnorm1(arr1, arr2))      # min
# # # print(tnorm2(arr1, arr2))      # product
# # # print(tnorm3(arr1, arr2))      # yager with p=3.0

# # print(tn.TNorm.list_available())
# ##################################################
# # check help
# print(tnorm2.help())

# # check create/from/to dict without param
# conf = tnorm2.to_dict()
# tnorm4 = tn.TNorm.from_dict(conf)

# print(tnorm4.help())

# # check create/from/to dict with param
# conf = tnorm3.to_dict()
# tnorm5 = tn.TNorm.from_dict(conf)

# print(tnorm5.help())

# a=tnorm3(.7,.3)
# a=tnorm2(arr1,arr2)
# a=tnorm2(arr3,arr4)
# print(a)

################################################################
# generate interim results for testint itfrs

data = sds.get_ITFRS_testing_testsets()

y = data[0]["y"]

Ay = (y[:, None] == y[None, :]).astype(float)
sim_matrix = data[0]["sim_matrix"]

a_sum_b = sim_matrix + Ay   
a_sum_b_minus_one = sim_matrix + Ay - 1.0 

a_mul_b = sim_matrix * Ay #elementwise multiplication
a_sum_b_minus_ab = a_sum_b - a_mul_b
two_minus_a_sum_b_minus_ab = 2.0 - a_sum_b_minus_ab

einstein_tn = a_mul_b/two_minus_a_sum_b_minus_ab

intrim_tn = a_mul_b/a_sum_b_minus_ab

p = 0.83
t1 = ((1.0 - sim_matrix)**p + (1.0 - Ay)**p)**(1.0/p)

print(t1)


# implication_vals = self.lb_implicator(self.similarity_matrix, label_mask)
# np.fill_diagonal(implication_vals, 1.0)

