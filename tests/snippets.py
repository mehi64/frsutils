import numpy as np

# a = np.array([ 
#     [[0.0, 0.0,0.0],[0.1, 0.46, 0.45],[0.63, 0.14, 0.20],[0.81, 0.16, 0.25],[0.90, 0.04, 0.01]],
#     [[0.1, 0.46, 0.45],[0.0, 0.0,0.0],[0.53, 0.60, 0.65],[0.71, 0.30, 0.20],[0.80, 0.50, 0.46]],
#     [[0.63, 0.14, 0.20],	[0.53, 0.60, 0.65],	[0.0, 0.0,0.0],	[0.18, 0.30, 0.45],	[0.27, 0.10, 0.19]],
#     [[0.81, 0.16, 0.25],	[0.71, 0.30, 0.20],	[0.18, 0.30, 0.45],	[0.0, 0.0,0.0],	[0.09, 0.20, 0.26]],
#     [[0.90, 0.04, 0.01],	[0.80, 0.50, 0.46],	[0.27, 0.10, 0.19],	[0.09, 0.20, 0.26],	[0.0, 0.0,0.0]]
# ])

# # b = 1.0 -a
# # print(b)
# ##################################################
# import FRsutils.core.owa_weights as oww

# w_inf = oww._owa_suprimum_weights_linear(8)
# print(w_inf)
# w_inf = oww._owa_suprimum_weights_linear(13)
# print(w_inf)

# import FRsutils.core.models.vqrs as VQRS_


from FRsutils.core.config_tnorm import TNormConfig
from FRsutils.core.config_similarity import SimilarityConfig
from FRsutils.core.config_implicator import ImplicatorConfig
from FRsutils.core.core_builders import build_tnorm, build_similarity, build_implicator
from FRsutils.core.models.config_fr_model import ITFRSConfig
from FRsutils.core.models.build_fr_model import build_itfrs
import tests.syntetic_data_for_tests as sdt


tnorm_cfg = TNormConfig(type='min')
sim_cfg = SimilarityConfig(type='gaussian', sigma=0.3)
impl_cfg = ImplicatorConfig(type='lukasiewicz')

tnorm = build_tnorm(tnorm_cfg)
similarity = build_similarity(sim_cfg)
implicator = build_implicator(impl_cfg)

ITFRSConfig_=ITFRSConfig(
    ub_tnorm=TNormConfig(type="min"),
    similarity=SimilarityConfig(type="gaussian", sigma=0.2),
    lb_implicator=ImplicatorConfig(type="lukasiewicz"),
    similarity_tnorm=TNormConfig(type="min")
)

X=np.array([[.1, .2],[.4, .2]])
y=np.array([1,0])

model = build_itfrs(ITFRSConfig_, X, y)
print(model)
