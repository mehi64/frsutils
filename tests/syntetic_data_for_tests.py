import numpy as np

class syntetic_dataset_factory:
    def __init__(self):
        pass

    def implicator_testing_data(self):
        """
        order of columns: a   b
        """
        
        a_b = np.array([
            # [2.10, 4.32],
            # [-0.20, -0.78],
            [0.73, 0.18],
            [0.18, 0.73],
            [0.88, 0.88],
            [0.91, 0.48],
            [1.00, 1.00],
            [0.00, 0.00]
        ])

        gaines_outputs = np.array([0.24657534246, 1.0, 1.0, 0.527472527, 1.0, 1.0])
        goedel_outputs= np.array([0.18, 1.0, 1.0, 0.48, 1.0, 1.0])
        luk_outputs = np.array([0.45, 1.00, 1.00, 0.57, 1.00, 1.00])
        kleene_dienes_outputs = np.array([0.27, 0.82, 0.88, 0.48, 1.00, 1.00])
        reichenbach_outputs = np.array([0.4014, 0.9514, 0.8944, 0.5268, 1.00, 1.00])

        data_dict = {"a_b" : a_b,
                     "gaines_outputs" : gaines_outputs,
                     "goedel_outputs": goedel_outputs,
                     "kleene_dienes_outputs" : kleene_dienes_outputs,
                     "luk_outputs" : luk_outputs,
                     "reichenbach_outputs" : reichenbach_outputs
                     }
        return data_dict
    
    def tnorm_scalar_testing_data(self):
        """
        order of columns: a   b
        """
        
        a_b = np.array([
            # [2.10, 4.32],
            # [-0.20, -0.78],
            [0.73, 0.18],
            [0.18, 0.73],
            [0.88, 0.88],
            [0.91, 0.48],
            [1.00, 1.00],
            [0.00, 0.00]
        ])

        minimum_outputs = np.array([0.18, 0.18, 0.88, 0.48, 1.0, 0.0])
        product_outputs= np.array([0.1314, 0.1314, 0.7744, 0.4368, 1.00, 0.00])
        luk_outputs = np.array([0.00, 0.00, 0.76, 0.39, 1.00, 0.00])

        data_dict = {"a_b" : a_b,
                     "minimum_outputs" : minimum_outputs,
                     "product_outputs": product_outputs
                     ,"luk_outputs" : luk_outputs
                     }
        return data_dict
    
    def tnorm_nxnx2_testing_dataset(self):
        """
        order of maps: nxn[0] mimics similarities therefore square, lower and upper
        triangles are mirrosed. Main diagonal is 1.0
        This is with binary masks matrix. Therefore tnorm min and product will act the same. SO we need another dataset
        """
        
        similarity_matrix = np.array([
        
            [1.0,     0.2673,  0.25456, 0.1197,  0.09504],
            [0.2673,  1.0,     0.0658,  0.1624,  0.054  ],
            [0.25456, 0.0658,  1.0,       0.3157,  0.53217],
            [0.1197,  0.1624,  0.3157,  1.0,     0.53872],
            [0.09504, 0.054,   0.53217, 0.53872, 1.0     ]
        ])

        label_mask = np.array([
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0]
        ])

        minimum_outputs = np.array([
            [1.0,     0.2673,  0.0,     0.1197,  0.0],
            [0.2673,  1.0,     0.0,     0.1624,  0.0],
            [0.0,     0.0,     1.0,     0.0,     0.53217],
            [0.1197,  0.1624,  0.0,     1.0,     0.0],
            [0.0,     0.0,     0.53217, 0.0,     1.0]
        ])
        product_outputs = np.array([
            [1.0,     0.2673,  0.0,     0.1197,  0.0],
            [0.2673,  1.0,     0.0,     0.1624,  0.0],
            [0.0,     0.0,     1.0,     0.0,     0.53217],
            [0.1197,  0.1624,  0.0,     1.0,     0.0],
            [0.0,     0.0,     0.53217, 0.0,     1.0]
        ])

        luk_outputs = np.array([
            [1.0,	    0.2673, 0.0,	0.1197,	0.0],
            [0.2673,	1.0,	0.0,	0.1624,	0.0],
            [0.0,	    0.0,	1.0,	0.0,	0.53217],
            [0.1197,	0.1624,	0.0,	1.0,	0.0],
            [0.0,	    0.0,	0.53217,0.0,	1.0]
        ])

        data_dict = {"similarity_matrix" : similarity_matrix,
                     "label_mask" : label_mask,
                     "minimum_outputs" : minimum_outputs,
                     "product_outputs": product_outputs
                     ,"luk_outputs" : luk_outputs
                     }
        return data_dict
    
    def similarity_testing_dataset(self):
        X = np.array([
            [0.10, 0.32, 0.48],
            [0.20, 0.78, 0.93],
            [0.73, 0.18, 0.28],
            [0.91, 0.48, 0.73],
            [1.00, 0.28, 0.47]
        ])

        sim_matrix_with_linear_similarity_product_tnorm = np.array([
            [1.0,     0.2673,  0.25456, 0.1197,  0.09504],
            [0.2673,  1.0,     0.0658,  0.1624,  0.054  ],
            [0.25456, 0.0658,  1.0,       0.3157,  0.53217],
            [0.1197,  0.1624,  0.3157,  1.0,     0.53872],
            [0.09504, 0.054,   0.53217, 0.53872, 1.0     ]
        ])

        sim_matrix_with_linear_similarity_minimum_tnorm = np.array([
            [1.00, 0.54, 0.37, 0.19, 0.10],
            [0.54, 1.00, 0.35, 0.29, 0.20],
            [0.37, 0.35, 1.00, 0.55, 0.73],
            [0.19, 0.29, 0.55, 1.00, 0.74],
            [0.10, 0.20, 0.73, 0.74, 1.00]
        ])

        sim_matrix_with_linear_similarity_luk_tnorm = np.array([
            [1.00,	0.00,	0.03,	0.00,	0.05],
            [0.00,	1.00,	0.00,	0.00,	0.00],
            [0.03,	0.00,	1.00,	0.07,	0.44],
            [0.00,	0.00,	0.07,	1.00,	0.45],
            [0.05,	0.00,	0.44,	0.45,	1.00]
        ])

        sim_matrix_with_Gaussian_similarity_product_tnorm = np.array([
            [1.0000,	0.6235,	0.6014,	0.4365,	0.4049],
            [0.6235,	1.0,	0.3059,	0.4935,	0.2932],
            [0.6014,	0.3059,	1.0,	0.6964,	0.8759],
            [0.4365,	0.4935,	0.6964,	1.0,	0.8791],
            [0.4049,	0.2932,	0.8759,	0.8791,	1.0]
        ])

        sim_matrix_with_Gaussian_similarity_minimum_tnorm = np.array([
            [1.,     0.79,   0.6427, 0.4815, 0.4057],
            [0.79,   1.,     0.6246, 0.5704, 0.4902],
            [0.6427, 0.6246, 1.,     0.7981, 0.922 ],
            [0.4815, 0.5704, 0.7981, 1.,     0.9275],
            [0.4057, 0.4902, 0.922,  0.9275, 1.    ]
        ])

        sim_matrix_with_Gaussian_similarity_luk_tnorm = np.array([
            [1.0000,	0.5770,	0.5775,	0.3862,	0.4038],
            [0.5770,	1.0000,	0.0256,	0.4314,	0.0371],
            [0.5775,	0.0256,	1.0000,	0.6673,	0.8715],
            [0.3862,	0.4314,	0.6673,	1.0000,	0.8749],
            [0.4038,	0.0371,	0.8715,	0.8749,	1.0000]
        ])

        data_dict = {"X" : X,
                     "sim_matrix_with_linear_similarity_product_tnorm" : sim_matrix_with_linear_similarity_product_tnorm,
                     "sim_matrix_with_linear_similarity_minimum_tnorm" : sim_matrix_with_linear_similarity_minimum_tnorm,
                     "sim_matrix_with_linear_similarity_luk_tnorm" : sim_matrix_with_linear_similarity_luk_tnorm,
                     "sim_matrix_with_Gaussian_similarity_product_tnorm" : sim_matrix_with_Gaussian_similarity_product_tnorm,
                     "sim_matrix_with_Gaussian_similarity_minimum_tnorm" : sim_matrix_with_Gaussian_similarity_minimum_tnorm,
                     "sim_matrix_with_Gaussian_similarity_luk_tnorm" : sim_matrix_with_Gaussian_similarity_luk_tnorm,
                     "sigma_for_Gaussian" : 0.67
                    }
        return data_dict
    
    def VQRS_testing_dataset(self):
        
        y = np.array([1, 1, 0, 1, 0])

        sim_matrix = np.array([
            [1.00, 0.54, 0.37, 0.19, 0.10],
            [0.54, 1.00, 0.35, 0.29, 0.20],
            [0.37, 0.35, 1.00, 0.55, 0.73],
            [0.19, 0.29, 0.55, 1.00, 0.74],
            [0.10, 0.20, 0.73, 0.74, 1.00]
        ])

        upper_bound = np.array([0.521049496528125,	0.50362976378912,	0.085078125,	0.015835966374605,	0.141019503319767628125])
        lower_bound = np.array([1.0,	1.0,	0.5582,	0.2344383779189888,	0.718538097101394872])

        data_dict = {"y" : y,
                     "sim_matrix" : sim_matrix,
                     "upper_bound" : upper_bound,
                     "lower_bound" : lower_bound,
                     "alpha_lower" : 0.1,
                     "beta_lower"  :0.6,
                     "alpha_upper" :0.2,
                     "beta_upper"  :1.0,
                     "fuzzy_quantifier": 'quadratic'
                    }
        return data_dict

    def ITFRS_testing_dataset(self):
        
        y = np.array([1, 1, 0, 1, 0])

        sim_matrix = np.array([
            [1.00, 0.54, 0.37, 0.19, 0.10],
            [0.54, 1.00, 0.35, 0.29, 0.20],
            [0.37, 0.35, 1.00, 0.55, 0.73],
            [0.19, 0.29, 0.55, 1.00, 0.74],
            [0.10, 0.20, 0.73, 0.74, 1.00]
        ])

        Reichenbach_lowerBound = np.array([0.63, 0.65, 0.45, 0.26, 0.26])
        KD_lowerBound = np.array([0.63, 0.65, 0.45, 0.26, 0.26])
        Luk_lowerBound = np.array([0.63, 0.65, 0.45, 0.26, 0.26])
        Goedel_lowerBound = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        Gaines_lowerBound = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        prod_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])
        min_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])

        data_dict = {"y" : y,
                     "sim_matrix" : sim_matrix,
                     "Reichenbach_lowerBound" : Reichenbach_lowerBound,
                     "KD_lowerBound" : KD_lowerBound,
                     "Luk_lowerBound" : Luk_lowerBound,
                     "Goedel_lowerBound" : Goedel_lowerBound,
                     "Gaines_lowerBound" : Gaines_lowerBound,
                     "prod_tn_upperBound" : prod_tn_upperBound,
                     "min_tn_upperBound" : min_tn_upperBound
                    }
        return data_dict
    
    def OWAFRS_testing_dataset(self):
        
        y = np.array([1, 1, 0, 1, 0])

        sim_matrix = np.array([
            [1.00, 0.54, 0.37, 0.19, 0.10],
            [0.54, 1.00, 0.35, 0.29, 0.20],
            [0.37, 0.35, 1.00, 0.55, 0.73],
            [0.19, 0.29, 0.55, 1.00, 0.74],
            [0.10, 0.20, 0.73, 0.74, 1.00]
        ])

        owa_linear_Reichenbach_lowerBound = np.array([0.822 , 0.8, 0.599, 0.539, 0.624])
        owa_linear_KD_lowerBound = np.array([0.822 , 0.8, 0.599, 0.539, 0.624])
        Luk_lowerBound = np.array([0.63, 0.65, 0.45, 0.26, 0.26])
        Goedel_lowerBound = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        Gaines_lowerBound = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        prod_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])
        min_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])

        data_dict = {"y" : y,
                     "sim_matrix" : sim_matrix,
                     "Reichenbach_lowerBound" : Reichenbach_lowerBound,
                     "KD_lowerBound" : KD_lowerBound,
                     "Luk_lowerBound" : Luk_lowerBound,
                     "Goedel_lowerBound" : Goedel_lowerBound,
                     "Gaines_lowerBound" : Gaines_lowerBound,
                     "prod_tn_upperBound" : prod_tn_upperBound,
                     "min_tn_upperBound" : min_tn_upperBound
                    }
        return data_dict

    def owa_weights_linear_testing_data(self):
        """
        owa weights
        """
        owa_infimum_weights_linear_len_5 = np.array([0.06666667, 0.13333333, 0.2, 0.26666667, 0.33333333])
        owa_infimum_weights_linear_len_10 = np.array([0.01818182, 0.03636364, 0.05454545, 0.07272727, 0.09090909, 0.10909091, 0.12727273, 0.14545455, 0.16363636, 0.18181818])
        
        owa_suprimum_weights_linear_len_8 = np.array([0.22222222, 0.19444444, 0.16666667, 0.13888889, 0.11111111, 0.08333333, 0.05555556, 0.02777778])
        owa_supriimum_weights_linear_len_13 = np.array([0.14285714, 0.13186813, 0.12087912, 0.10989011, 0.0989011,  0.08791209, 0.07692308, 0.06593407, 0.05494505, 0.04395604, 0.03296703, 0.02197802, 0.01098901])
        

        data_dict = {"owa_infimum_weights_linear_len_5" : owa_infimum_weights_linear_len_5,
                     "owa_infimum_weights_linear_len_10" : owa_infimum_weights_linear_len_10,
                     "owa_suprimum_weights_linear_len_8":owa_suprimum_weights_linear_len_8,
                     "owa_suprimum_weights_linear_len_13":owa_supriimum_weights_linear_len_13}
        return data_dict