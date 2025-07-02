import numpy as np
from FRsutils.core.similarities import Similarity, calculate_similarity_matrix
from FRsutils.core.tnorms import TNorm
from FRsutils.core.implicators import Implicator
from FRsutils.core.models.owafrs import OWAFRS
from FRsutils.core.owa_weights import OWAWeights
from FRsutils.utils.logger.logger_util import get_logger
import tests.synthetic_data_store as sdf
from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel as FRMODEL

logger = get_logger()

data_synthteic = sdf.get_OWAFRS_testing_testsets()[0]

TSTowaFRS = data_synthteic
sim_matrix = TSTowaFRS['sim_matrix']
labels = TSTowaFRS['y']

expected = TSTowaFRS["expected"]["owa_linear"]



# sim_matrix= np.array([
#     [1.00, 0.54, 0.37, 0.19, 0.10],
#     [0.54, 1.00, 0.35, 0.29, 0.20],
#     [0.37, 0.35, 1.00, 0.55, 0.73],
#     [0.19, 0.29, 0.55, 1.00, 0.74],
#     [0.10, 0.20, 0.73, 0.74, 1.00]
# ])
# labels = np.array([1, 1, 0, 1, 0])

# tnrm=TNorm.create("prod")
# similarity_func = Similarity.create("gaussian", tnrm, sigma=0.3)

# sim_matrix2 = calculate_similarity_matrix(X, similarity_func, tnrm)

# Create OWAFRS model
tnorm = TNorm.create("hamacher", p=0.83)
implicator = Implicator.create("weber")

lb_owa = OWAWeights.create("linear")
ub_owa = OWAWeights.create("linear")

model = OWAFRS(similarity_matrix=sim_matrix,
              labels=labels,
              ub_tnorm=tnorm,
              lb_implicator=implicator,
              lb_owa_method=lb_owa,
              ub_owa_method=ub_owa,
              logger=logger)

# # a = model.get_class('owafrs')

upper = model.upper_approximation()
# lower = model.lower_approximation()

print(upper)

# d = model.to_dict(include_data= True)
# obj = OWAFRS.from_dict(d)

# aa = model.describe_params_detailed()


# #########################################################

# config_without_sim_label = {
#     # 'similarity_matrix': sim_matrix, 
#     #              'labels': y, 
#                 'ub_owa_method_name':'linear',
#                 'lb_owa_method_name':'linear',
#                 'base':2.0,
#                  'ub_tnorm_name': 'minimum', 
#                  'lb_implicator_name': 'luk',
#                  'logger':logger,
#                  'kneighbors':5}

# mdl2 = FRMODEL.get_class("owafrs").from_config(sim_matrix,labels,**config_without_sim_label)


# print(1)
