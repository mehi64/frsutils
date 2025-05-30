import numpy as np

import tests.syntetic_data_for_tests as sds


import FRsutils.core.similarities as sim
import FRsutils.core.tnorms as tn


def test_compute_similarity_matrix_with_linear_similarity_product_tnorm():
    dsm = sds.syntetic_dataset_factory()
    data_dict = dsm.similarity_testing_dataset()
    X = data_dict["X"]
    expected = data_dict["sim_matrix_with_linear_similarity_product_tnorm"]

    tnrm = tn.ProductTNorm()
    sim_f = sim.LinearSimilarity()
    sim_matrix = sim.calculate_similarity_matrix(X, similarity_func=sim_f, tnorm=tnrm)
    assert sim_matrix.shape == (5, 5), "dimension mismatch"
    assert (0.0 <= sim_matrix).all() and (sim_matrix <= 1.0).all(), "similarity matrix values are not normalized"
    closeness = np.isclose(sim_matrix, expected)
    assert np.all(closeness), "outputs are not the expected values"


def test_compute_similarity_matrix_with_linear_similarity_minimum_tnorm():
    dsm = sds.syntetic_dataset_factory()
    data_dict = dsm.similarity_testing_dataset()
    X = data_dict["X"]
    expected = data_dict["sim_matrix_with_linear_similarity_minimum_tnorm"]

    tnrm = tn.MinTNorm()
    sim_f = sim.LinearSimilarity()
    sim_matrix = sim.calculate_similarity_matrix(X, similarity_func=sim_f, tnorm=tnrm)
    assert sim_matrix.shape == (5, 5), "dimension mismatch"
    assert (0.0 <= sim_matrix).all() and (sim_matrix <= 1.0).all(), "similarity matrix values are not normalized"
    closeness = np.isclose(sim_matrix, expected)
    assert np.all(closeness), "outputs are not the expected values"

def test_compute_similarity_matrix_with_linear_similarity_luk_tnorm():
    dsm = sds.syntetic_dataset_factory()
    data_dict = dsm.similarity_testing_dataset()
    X = data_dict["X"]
    expected = data_dict["sim_matrix_with_linear_similarity_luk_tnorm"]

    tnrm = tn.LukasiewiczTNorm()
    sim_f = sim.LinearSimilarity()
    sim_matrix = sim.calculate_similarity_matrix(X, similarity_func=sim_f, tnorm=tnrm)
    assert sim_matrix.shape == (5, 5), "dimension mismatch"
    assert (0.0 <= sim_matrix).all() and (sim_matrix <= 1.0).all(), "similarity matrix values are not normalized"
    closeness = np.isclose(sim_matrix, expected)
    assert np.all(closeness), "outputs are not the expected values"

def test_compute_similarity_matrix_with_Gaussian_similarity_product_tnorm():
    dsm = sds.syntetic_dataset_factory()
    data_dict = dsm.similarity_testing_dataset()
    X = data_dict["X"]
    expected = data_dict["sim_matrix_with_Gaussian_similarity_product_tnorm"]

    tnrm = tn.ProductTNorm()
    sim_f = sim.GaussianSimilarity(sigma=0.67)
    sim_matrix = sim.calculate_similarity_matrix(X, similarity_func=sim_f, tnorm=tnrm)
    assert sim_matrix.shape == (5, 5), "dimension mismatch"
    assert (0.0 <= sim_matrix).all() and (sim_matrix <= 1.0).all(), "similarity matrix values are not normalized"
    closeness = np.isclose(sim_matrix, expected, rtol=0, atol=1e-4)
    assert np.all(closeness), "outputs are not the expected values"


def test_compute_similarity_matrix_with_Gaussian_similarity_minimum_tnorm():
    dsm = sds.syntetic_dataset_factory()
    data_dict = dsm.similarity_testing_dataset()
    X = data_dict["X"]
    expected = data_dict["sim_matrix_with_Gaussian_similarity_minimum_tnorm"]

    tnrm = tn.MinTNorm()
    sim_f = sim.GaussianSimilarity(sigma=0.67)
    sim_matrix = sim.calculate_similarity_matrix(X, similarity_func=sim_f, tnorm=tnrm)
    assert sim_matrix.shape == (5, 5), "dimension mismatch"
    assert (0.0 <= sim_matrix).all() and (sim_matrix <= 1.0).all(), "similarity matrix values are not normalized"
    closeness = np.isclose(sim_matrix, expected, rtol=0, atol=1e-4)
    assert np.all(closeness), "outputs are not the expected values"

def test_compute_similarity_matrix_with_Gaussian_similarity_luk_tnorm():
    dsm = sds.syntetic_dataset_factory()
    data_dict = dsm.similarity_testing_dataset()
    X = data_dict["X"]
    expected = data_dict["sim_matrix_with_Gaussian_similarity_luk_tnorm"]

    tnrm = tn.LukasiewiczTNorm()
    sim_f = sim.GaussianSimilarity(sigma=0.67)
    sim_matrix = sim.calculate_similarity_matrix(X, similarity_func=sim_f, tnorm=tnrm)
    assert sim_matrix.shape == (5, 5), "dimension mismatch"
    assert (0.0 <= sim_matrix).all() and (sim_matrix <= 1.0).all(), "similarity matrix values are not normalized"
    closeness = np.isclose(sim_matrix, expected, rtol=0, atol=1e-3)
    assert np.all(closeness), "outputs are not the expected values"
