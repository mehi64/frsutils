import FRsutils.core.similarities as sim
import FRsutils.core.tnorms as tn
import tests.synthetic_data_store as sds
import numpy as np

tnrm0 = tn.TNorm.create('min')
dic0 = tnrm0.to_dict()
tnrm1 = tn.TNorm.from_dict(dic0)
nme = tnrm0.name
sim1 = sim.Similarity.create('linear', strict=False, sigma=0.67)
params = sim1.get_params_detailed()
hlp =sim1.help()
print(hlp)
nme = sim1.name
dic1 = sim1.to_dict()
similarity_instance = sim.Similarity.from_dict(dic1)

ds = sds.get_similarity_testing_testsets()

X = ds[0]['X']
vals = ds[0]['expected']

sim_mat = sim.calculate_similarity_matrix(
    X,
    sim1,
    tnrm0)

print(sim_mat)


