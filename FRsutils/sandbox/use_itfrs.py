import numpy as np
from FRsutils.core.similarities import Similarity, calculate_similarity_matrix
from FRsutils.core.tnorms import TNorm
from FRsutils.core.implicators import Implicator
from FRsutils.core.models.itfrs import ITFRS
from FRsutils.utils.logger.logger_util import get_logger
import tests.synthetic_data_store as sdf
from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel as FRMODEL


data_synthteic = sdf.get_ITFRS_testing_testsets()[0]

TSTITFRS = data_synthteic
sim_matrix = TSTITFRS['sim_matrix']
y = TSTITFRS['y']

# X = np.array([
#     [0.10, 0.32, 0.48],
#     [0.20, 0.78, 0.93],
#     [0.73, 0.18, 0.28],
#     [0.91, 0.48, 0.73],
#     [1.00, 0.28, 0.47]
# ])
# labels = np.array([1, 1, 0, 1, 0])

# tnrm=TNorm.create("minimum")

# # Create Gaussian similarity with minimum tnorm
# similarity_func = SimilarityFunction.create("gaussian", tnrm, sigma=0.3)

# sim_matrix2 = calculate_similarity_matrix(X, similarity_func, tnrm)




# Create ITFRS model with product tnorm and gaines implicator
tnorm = TNorm.create("yager", p=0.83)
implicator = Implicator.create("fodor")

logger = get_logger()

model = ITFRS(similarity_matrix=sim_matrix,
              labels=y,
              ub_tnorm=tnorm,
              lb_implicator=implicator,
              logger=logger)

aa = model.describe_params_detailed()

upper = model.upper_approximation()
lower = model.lower_approximation()

config_without_sim_label = {
    # 'similarity_matrix': sim_matrix, 
    #              'labels': y, 
                 'ub_tnorm_name': 'minimum', 
                 'lb_implicator_name': 'luk',
                 'logger':logger,
                 'kneighbors':5}

mdl2 = FRMODEL.get_class("itfrs").from_config(sim_matrix,y,config=config_without_sim_label)

frmodel = FRMODEL.create(name='itfrs', strict=False, **config_without_sim_label)

upper1 = frmodel.upper_approximation()
lower1 = frmodel.lower_approximation()

conf = frmodel.to_dict(include_data=True)
mdl = FRMODEL.get_class("itfrs").from_config(conf)

conf2 = frmodel.to_dict(include_data=False)
mdl2 = FRMODEL.get_class("itfrs").from_config(conf2,sim_matrix,y)

fr_model3 = FRMODEL.get_class("itfrs").from_dict(conf2,sim_matrix,y)
fr_model4 = FRMODEL.get_class("itfrs").from_dict(conf)

upper2 = mdl.upper_approximation()
lower2 = mdl.lower_approximation()

upper4 = fr_model4.upper_approximation()
lower4 = fr_model4.lower_approximation()

# print("tnorm:", tnorm.name)
# print("implicator:", implicator.name)
print("Lower Approximation:", lower4)
print("Lower Approximation:", lower)
print("Lower Approximation:", lower2)
print("Upper Approximation:", upper)
print("Upper Approximation:", upper2)
print("Upper Approximation:", upper4)

print("Done")

#############################################################


