

#######################################################################
# # t-norms
import tests.test_tnorms as tt

tt.test_tn_minimum_scalar_values()
tt.test_tn_product_scalar_values()
tt.test_tn_luk_scalar_values()
tt.test_tn_minimum_nxnx2_map_values()
tt.test_tn_product_nxnx2_map_values()
tt.test_tn_luk_nxnx2_map_values()


######################################################################
## implicators
import tests.test_implicators as timp

timp.test_goedel_implicator_outputs()
timp.test_gaines_implicator_outputs()
timp.test_luk_implicator_outputs()
timp.test_kd_implicator_outputs()
timp.test_reichenbach_implicator_outputs()
######################################################################
# similarities
import tests.test_similarities as ts
ts.test_compute_similarity_matrix_with_linear_similarity_product_tnorm()
ts.test_compute_similarity_matrix_with_linear_similarity_minimum_tnorm()
ts.test_compute_similarity_matrix_with_linear_similarity_luk_tnorm()
ts.test_compute_similarity_matrix_with_Gaussian_similarity_product_tnorm()
ts.test_compute_similarity_matrix_with_Gaussian_similarity_minimum_tnorm()
ts.test_compute_similarity_matrix_with_Gaussian_similarity_luk_tnorm()

##################################################################
# # itfrs
import tests.test_itfrs as ti
ti.test_itfrs_approximations_reichenbach_imp_product_tnorm()
ti.test_itfrs_approximations_KD_imp_product_tnorm()
ti.test_itfrs_approximations_Luk_imp_product_tnorm()
ti.test_itfrs_approximations_Goedel_imp_product_tnorm()
ti.test_itfrs_approximations_Gaines_imp_product_tnorm()

##################################################################
# # # owa_weights
# import owa_weights as ow

# ow._owa_suprimum_weights_linear(5)

##################################################################
# owafrs
import tests.test_owafrs as to
to.test_owafrs_approximations_reichenbach_imp_product_tnorm()


##################################################################
# vqrs
import tests.test_vqrs as tv
# tv.test_vqrs_lower_upper_approximations_quadratic_fuzzyquantifier()

##################################################################
# FRSMOTE
import tests.test_FRSMOTE as tfrsmote
tfrsmote.test1()





