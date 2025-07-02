"""
Synthetic Test Datasets for Fuzzy Components

Provides structured test datasets for verifying T-norms, Implicators, and other fuzzy components.
"""

import numpy as np

def get_tnorm_call_testsets():
    return [
        {
            "name": "basic_tnorms_DS_1",
            "a_b": np.array([
                [0.73, 0.18],
                [0.18, 0.73],
                [0.88, 0.88],
                [0.91, 0.48],
                [1.00, 1.00],
                [0.00, 0.00],
                [1.00, 0.65],
                [0.37, 1.00]
            ]),
            "expected": {
                "minimum": np.array([0.18, 0.18, 0.88, 0.48, 1.0, 0.0, 0.65, 0.37]),
                "product": np.array([0.1314, 0.1314, 0.7744, 0.4368, 1.00, 0.00, 0.65, 0.37]),
                "lukasiewicz": np.array([0.0, 0.0, 0.76, 0.39, 1.00, 0.00, 0.65, 0.37]),
                "drastic": np.array([0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.65, 0.37]),
                "hamacher": np.array([0.168764, 0.168764, 0.785714, 0.458246, 1.00, 0.00, 0.65, 0.37]),
                "einstein": np.array([0.107581, 0.107581, 0.763407, 0.417271, 1.00, 0.00, 0.65, 0.37]),
                "nilpotent": np.array([0.00, 0.00, 0.88, 0.48, 1.00, 0.00, 0.65, 0.37]),
                "yager_p=0.835": np.array([0.00, 0.00, 0.724771, 0.332934, 1.00, 0.00, 0.65, 0.37]),
                "yager_p=5.0": np.array([0.179366244, 0.179366244, 0.8621561974, 0.4799838489, 1.00, 0.00, 0.65, 0.37])
            }
        }
    ]

def get_tnorm_reduce_testsets():
    return [
        {
            "name": "tnorm_reduce",
            "similarity_matrix": np.array([
            [1.0,     0.2673,  0.25456, 0.1197,  0.09504],
            [0.2673,  1.0,     0.0658,  0.1624,  0.054  ],
            [0.25456, 0.0658,  1.0,     0.3157,  0.53217],
            [0.1197,  0.1624,  0.3157,  1.0,     0.53872],
            [0.09504, 0.054,   0.53217, 0.53872, 1.0     ]
        ]),
            "label_mask" : np.array([
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0]
        ]),
            "expected": 
            {
                "minimum_outputs": np.array([
                [1.0,     0.2673,  0.0,     0.1197,  0.0],
                [0.2673,  1.0,     0.0,     0.1624,  0.0],
                [0.0,     0.0,     1.0,     0.0,     0.53217],
                [0.1197,  0.1624,  0.0,     1.0,     0.0],
                [0.0,     0.0,     0.53217, 0.0,     1.0]]),
                
                "product_outputs": np.array([
                [1.0,     0.2673,  0.0,     0.1197,  0.0],
                [0.2673,  1.0,     0.0,     0.1624,  0.0],
                [0.0,     0.0,     1.0,     0.0,     0.53217],
                [0.1197,  0.1624,  0.0,     1.0,     0.0],
                [0.0,     0.0,     0.53217, 0.0,     1.0]]),
            
                "luk_outputs" : np.array([
                [1.0,	    0.2673, 0.0,	0.1197,	0.0],
                [0.2673,	1.0,	0.0,	0.1624,	0.0],
                [0.0,	    0.0,	1.0,	0.0,	0.53217],
                [0.1197,	0.1624,	0.0,	1.0,	0.0],
                [0.0,	    0.0,	0.53217,0.0,	1.0]])
            }
        }
    ]

def get_implicator_scalar_testsets():
    return [
        {
            "name": "basic_implicators",
            "a_b": np.array([
                [0.73, 0.18],
                [0.18, 0.73],
                [0.88, 0.88],
                [0.91, 0.48],
                [1.00, 1.00],
                [0.00, 0.00],
                [0.00, 1.00],
                [1.00, 0.00],
                [1.00, 0.65],
                [0.65, 1.00],
                [0.55, 0.00],
                [0.00, 0.55],
            ]),

            "expected": {
                "goedel": np.array([0.18, 1.0, 1.0, 0.48, 1.0, 1.0, 1.00, 0.00, 0.65, 1.00, 0.00, 1.00]),
                "lukasiewicz": np.array([0.45, 1.00, 1.00, 0.57, 1.00, 1.00, 1.00, 0.00, 0.65, 1.00, 0.45, 1.00]),
                "kleenedienes": np.array([0.27, 0.82, 0.88, 0.48, 1.00, 1.00, 1.00, 0.00, 0.65, 1.00, 0.45, 1.00]),
                "reichenbach": np.array([0.4014, 0.9514, 0.8944, 0.5268, 1.00, 1.00, 1.00, 0.00, 0.65, 1.00, 0.45, 1.00]),
                "goguen": np.array([0.246575, 1.00, 1.00, 0.527473, 1.00, 1.00, 1.00, 0.00, 0.65, 1.00, 0.00, 1.00]),
                "rescher": np.array([0.00, 1.00, 1.00, 0.00, 1.00, 1.00, 1.00, 0.00, 0.00, 1.00, 0.00, 1.00]),
                "yager": np.array([0.285989, 0.944927, 0.893603, 0.512778, 1.00, 1.00, 1.00, 0.00, 0.65, 1.00, 0.00, 1.00]),
                "weber": np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.00, 0.65, 1.00, 1.00, 1.00]),
                "fodor": np.array([0.27, 1.00, 1.00, 0.48, 1.00, 1.00, 1.00, 0.00, 0.65, 1.00, 0.45, 1.00])
            }
        }
    ]

def owa_weights_testing_testsets():
    """
    owa weights
    """

    linear_asc_len_5  = np.array([0.06666667, 0.13333333, 0.2, 0.26666667, 0.33333333])
    linear_asc_len_8  = np.array([0.02777778, 0.05555556, 0.08333333, 0.11111111, 0.13888889, 0.16666667, 0.19444444, 0.22222222])
    linear_asc_len_10 = np.array([0.01818182, 0.03636364, 0.05454545, 0.07272727, 0.09090909, 0.10909091, 0.12727273, 0.14545455, 0.16363636, 0.18181818])
    linear_asc_len_13 = np.array([0.01098901, 0.02197802, 0.03296703, 0.04395604, 0.05494505, 0.06593407, 0.07692308, 0.08791209, 0.0989011, 0.10989011, 0.12087912, 0.13186813, 0.14285714])
        
    exp_asc_len_5_set1  = np.array([0.03225806, 0.06451613, 0.12903226, 0.25806452, 0.51612903])    
    exp_asc_len_8_set1  = np.array([0.00392157, 0.00784314, 0.01568627, 0.03137255, 0.0627451, 0.1254902, 0.25098039, 0.50196078])    
    exp_asc_len_10_set1  = np.array([0.00097752, 0.00195503, 0.00391007, 0.00782014, 0.01564027, 0.03128055, 0.06256109, 0.12512219, 0.25024438, 0.50048876])    
    exp_asc_len_13_set1  = np.array([0.00012209, 0.00024417, 0.00048834, 0.00097668, 0.00195336, 0.00390673, 0.00781345, 0.01562691, 0.03125382, 0.06250763, 0.12501526, 0.25003052, 0.50006104])    
    
    harmonic_asc_len_5  = np.array([0.08759124, 0.10948905, 0.1459854, 0.2189781, 0.4379562])    
    harmonic_asc_len_8  = np.array([0.04599212, 0.05256242, 0.06132282, 0.07358739, 0.09198423, 0.12264564, 0.18396846, 0.36793693])    
    harmonic_asc_len_10  = np.array([0.03414172, 0.03793524, 0.04267714, 0.04877388, 0.05690286, 0.06828343, 0.08535429, 0.11380572, 0.17070858, 0.34141715])    
    harmonic_asc_len_13  = np.array([0.02418863, 0.02620435, 0.02858656, 0.03144522, 0.03493913, 0.03930652, 0.04492174, 0.0524087, 0.06289044, 0.07861305, 0.10481739, 0.15722609, 0.31445218])    
    
    log_asc_len_5  = np.array([0.10535351, 0.16698136, 0.21070701, 0.24462326, 0.27233486])    
    log_asc_len_8  = np.array([0.05414439, 0.08581683, 0.10828879, 0.12571939, 0.13996123, 0.15200253, 0.16243318, 0.17163367])    
    log_asc_len_10  = np.array([0.03960319, 0.06276957, 0.07920638, 0.09195575, 0.10237275, 0.1111802, 0.11880956, 0.12553913, 0.13155894, 0.13700452])    
    log_asc_len_13  = np.array([0.02751543, 0.04361092, 0.05503085, 0.06388884, 0.07112634, 0.07724557, 0.08254628, 0.08722184, 0.09140427, 0.09518773, 0.09864177, 0.10181917, 0.10476099])    
    
    exp_asc_len_5_set2  = np.array([0.00826446, 0.02479339, 0.07438017, 0.2231405, 0.66942149])    
    exp_asc_len_8_set2  = np.array([0.00030488, 0.00091463, 0.0027439, 0.00823171, 0.02469512, 0.07408537, 0.2222561, 0.66676829])    
    exp_asc_len_10_set2  = np.array([0.00003387, 0.00010161, 0.00030484, 0.00091451, 0.00274353, 0.00823059, 0.02469178, 0.07407533, 0.22222599, 0.66667796])    
    exp_asc_len_13_set2  = np.array([0.00000125, 0.00000376, 0.00001129, 0.00003387, 0.00010161, 0.00030483, 0.0009145, 0.00274349, 0.00823046, 0.02469137, 0.07407412, 0.22222236, 0.66666708])    

 
    return [
    {
        "exp": {
            "dataset_1":
            {
                "base":2.0,
                "asc_OWA":{
                    "len_5": exp_asc_len_5_set1,
                    "len_8": exp_asc_len_8_set1,
                    "len_10": exp_asc_len_10_set1,
                    "len_13": exp_asc_len_13_set1
                },
                "desc_OWA":{
                    "len_5": exp_asc_len_5_set1[::-1],
                    "len_8": exp_asc_len_8_set1[::-1],
                    "len_10": exp_asc_len_10_set1[::-1],
                    "len_13": exp_asc_len_13_set1[::-1]
                }
            },
            "dataset_2":
            {
                "base":3.0,
                "asc_OWA":{
                    "len_5": exp_asc_len_5_set2,
                    "len_8": exp_asc_len_8_set2,
                    "len_10": exp_asc_len_10_set2,
                    "len_13": exp_asc_len_13_set2
                },
                "desc_OWA":{
                    "len_5": exp_asc_len_5_set2[::-1],
                    "len_8": exp_asc_len_8_set2[::-1],
                    "len_10": exp_asc_len_10_set2[::-1],
                    "len_13": exp_asc_len_13_set2[::-1]
                }
            }
            
        },
        "linear": {
            "asc_OWA":{
                "len_5": linear_asc_len_5,
                "len_8": linear_asc_len_8,
                "len_10": linear_asc_len_10,
                "len_13": linear_asc_len_13
            },
            "desc_OWA":{
                "len_5": linear_asc_len_5[::-1],
                "len_8": linear_asc_len_8[::-1],
                "len_10": linear_asc_len_10[::-1],
                "len_13": linear_asc_len_13[::-1]
            }
        },
        "harmonic": {
            "asc_OWA":{
                "len_5": harmonic_asc_len_5,
                "len_8": harmonic_asc_len_8,
                "len_10": harmonic_asc_len_10,
                "len_13": harmonic_asc_len_13
            },
            "desc_OWA":{
                "len_5": harmonic_asc_len_5[::-1],
                "len_8": harmonic_asc_len_8[::-1],
                "len_10": harmonic_asc_len_10[::-1],
                "len_13": harmonic_asc_len_13[::-1]
            }
        },
        "log": {
            "asc_OWA":{
                "len_5": log_asc_len_5,
                "len_8": log_asc_len_8,
                "len_10": log_asc_len_10,
                "len_13": log_asc_len_13
            },
            "desc_OWA":{
                "len_5": log_asc_len_5[::-1],
                "len_8": log_asc_len_8[::-1],
                "len_10": log_asc_len_10[::-1],
                "len_13": log_asc_len_13[::-1]
            }
        }
    }
    ]

def get_similarity_testing_testsets():
    return [
        {
            "name": "basic_similarity",
            "X": np.array([
            [0.10, 0.32, 0.48],
            [0.20, 0.78, 0.93],
            [0.73, 0.18, 0.28],
            [0.91, 0.48, 0.73],
            [1.00, 0.28, 0.47]
            ]),
            "sigma_for_gaussian_similarity" : 0.67,

            "expected": 
            {
                "sim_matrix_with_linear_similarity_product_tnorm": np.array([
                [1.0,     0.2673,  0.25456, 0.1197,  0.09504],
                [0.2673,  1.0,     0.0658,  0.1624,  0.054  ],
                [0.25456, 0.0658,  1.0,       0.3157,  0.53217],
                [0.1197,  0.1624,  0.3157,  1.0,     0.53872],
                [0.09504, 0.054,   0.53217, 0.53872, 1.0     ]
            ]),
                "sim_matrix_with_linear_similarity_minimum_tnorm" : np.array([
                [1.00, 0.54, 0.37, 0.19, 0.10],
                [0.54, 1.00, 0.35, 0.29, 0.20],
                [0.37, 0.35, 1.00, 0.55, 0.73],
                [0.19, 0.29, 0.55, 1.00, 0.74],
                [0.10, 0.20, 0.73, 0.74, 1.00]
            ]),
                "sim_matrix_with_linear_similarity_luk_tnorm" : np.array([
                [1.00,	0.00,	0.03,	0.00,	0.05],
                [0.00,	1.00,	0.00,	0.00,	0.00],
                [0.03,	0.00,	1.00,	0.07,	0.44],
                [0.00,	0.00,	0.07,	1.00,	0.45],
                [0.05,	0.00,	0.44,	0.45,	1.00]
            ]),

                "sim_matrix_with_gaussian_similarity_product_tnorm" : np.array([
                [1.0000,	0.6235,	0.6014,	0.4365,	0.4049],
                [0.6235,	1.0,	0.3059,	0.4935,	0.2932],
                [0.6014,	0.3059,	1.0,	0.6964,	0.8759],
                [0.4365,	0.4935,	0.6964,	1.0,	0.8791],
                [0.4049,	0.2932,	0.8759,	0.8791,	1.0]
            ])

                ,"sim_matrix_with_gaussian_similarity_minimum_tnorm" : np.array([
                [1.,     0.79,   0.6427, 0.4815, 0.4057],
                [0.79,   1.,     0.6246, 0.5704, 0.4902],
                [0.6427, 0.6246, 1.,     0.7981, 0.922 ],
                [0.4815, 0.5704, 0.7981, 1.,     0.9275],
                [0.4057, 0.4902, 0.922,  0.9275, 1.    ]
            ])

                ,"sim_matrix_with_gaussian_similarity_luk_tnorm" : np.array([
                [1.0000,	0.5770,	0.5775,	0.3862,	0.4038],
                [0.5770,	1.0000,	0.0256,	0.4314,	0.0371],
                [0.5775,	0.0256,	1.0000,	0.6673,	0.8715],
                [0.3862,	0.4314,	0.6673,	1.0000,	0.8749],
                [0.4038,	0.0371,	0.8715,	0.8749,	1.0000]
            ])
            }
        }
    ]

def get_ITFRS_testing_testsets():
    Reichenbach_lowerBound = np.array([0.63, 0.65, 0.45, 0.26, 0.26])
    KD_lowerBound = np.array([0.63, 0.65, 0.45, 0.26, 0.26])
    Luk_lowerBound = np.array([0.63, 0.65, 0.45, 0.26, 0.26])
    Fodor_lowerBound = np.array([0.63, 0.65, 0.45, 0.26, 0.26])

    Goedel_lowerBound = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    Goguen_lowerBound = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    Rescher_lowerBound = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    Yager_lowerBound = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    # Gaines_lowerBound = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    Weber_lowerBound = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    

    prod_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])
    min_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])
    einstein_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])
    luk_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])
    drastic_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])
    nilpotent_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])
    hamacher_tn_upperBound = np.array([0.54, 0.54, 0.73, 0.29, 0.73])
    yager_tn_upperBound_p_0_83 = np.array([0.54, 0.54, 0.73, 0.29, 0.73])
    
    return [
        {
            "name": "itfrs",
            "y" : np.array([1, 1, 0, 1, 0]),

            "sim_matrix": np.array([
            [1.00, 0.54, 0.37, 0.19, 0.10],
            [0.54, 1.00, 0.35, 0.29, 0.20],
            [0.37, 0.35, 1.00, 0.55, 0.73],
            [0.19, 0.29, 0.55, 1.00, 0.74],
            [0.10, 0.20, 0.73, 0.74, 1.00]
        ]),

            "expected": 
            {
                "Reichenbach_lowerBound" : Reichenbach_lowerBound,
                "KD_lowerBound" : KD_lowerBound,
                "Luk_lowerBound" : Luk_lowerBound,
                "Goedel_lowerBound" : Goedel_lowerBound,
                "Goguen_lowerBound" : Goguen_lowerBound,
                "Rescher_lowerBound" : Rescher_lowerBound,
                "Weber_lowerBound" : Weber_lowerBound,
                "Fodor_lowerBound" : Fodor_lowerBound,
                "Yager_lowerBound" : Yager_lowerBound,
                # "Gaines_lowerBound" : Gaines_lowerBound,
                "prod_tn_upperBound" : prod_tn_upperBound,
                "min_tn_upperBound" : min_tn_upperBound,
                "einstein_tn_upperBound" : einstein_tn_upperBound,
                "luk_tn_upperBound" : luk_tn_upperBound,
                "drastic_tn_upperBound" : drastic_tn_upperBound,
                "nilpotent_tn_upperBound" : nilpotent_tn_upperBound,
                "hamacher_tn_upperBound" : hamacher_tn_upperBound,
                "yager_tn_upperBound_p_0_83" : yager_tn_upperBound_p_0_83
            }
        }
    ]

def get_VQRS_testing_testsets():
    
    return [
        {
            "name": "vqrs",
            "y" : np.array([1, 1, 0, 1, 0]),
            "sim_matrix": np.array([
            [1.00, 0.54, 0.37, 0.19, 0.10],
            [0.54, 1.00, 0.35, 0.29, 0.20],
            [0.37, 0.35, 1.00, 0.55, 0.73],
            [0.19, 0.29, 0.55, 1.00, 0.74],
            [0.10, 0.20, 0.73, 0.74, 1.00]
        ]),
            "alpha_lower" : 0.1,
            "beta_lower"  :0.6,
            "alpha_upper" :0.2,
            "beta_upper"  :1.0,

            "expected": 
            {
                "quadratic_fuzzy_quantifier":
                {
                    "upper_bound" : np.array([0.5206163194,	0.5036166362,	0.085078125,	0.015835966374605,	0.141019503319767628125]),
                    "lower_bound" : np.array([1.0,	1.0,	0.5582,	0.2344383779189888,	0.718538097101394872]),
                },
                "linear_fuzzy_quantifier":
                {
                    "upper_bound" : np.array([0.51041625, 0.5018116, 0.20625, 0.08898305, 0.2655367238]),
                    "lower_bound" : np.array([1.0, 1.0, 0.53, 0.34237288, 0.624858758])
                }
            }
        }
    ]

def get_OWAFRS_testing_testsets():
    
    reichenbach_lowerBound = np.array([0.822 , 0.8, 0.599, 0.539, 0.624])
    kd_lowerBound = np.array([0.822 , 0.8, 0.599, 0.539, 0.624])
    fodor_lowerBound = np.array([0.822 , 0.8, 0.599, 0.539, 0.624])
    luk_lowerBound = np.array([0.822 , 0.8, 0.599, 0.539, 0.624])
    
    goedel_lowerBound = np.array([0.3, 0.3, 0.1, 0.3, 0.1])
    # Gaines_lowerBound = np.array()
    goguen_lowerBound = np.array([0.3, 0.3, 0.1, 0.3, 0.1])
    rescher_lowerBound = np.array([0.3, 0.3, 0.1, 0.3, 0.1])
    yager_lowerBound = np.array([0.3, 0.3, 0.1, 0.3, 0.1])

    weber_lowerBound = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    

    prod_tn_upperBound = np.array([0.273, 0.303, 0.292, 0.173, 0.292])
    min_tn_upperBound = np.array([0.273, 0.303, 0.292, 0.173, 0.292])
    einstein_tn_upperBound = np.array([0.273, 0.303, 0.292, 0.173, 0.292])
    luk_tn_upperBound = np.array([0.273, 0.303, 0.292, 0.173, 0.292])
    drastic_tn_upperBound = np.array([0.273, 0.303, 0.292, 0.173, 0.292])
    nilpotent_tn_upperBound = np.array([0.273, 0.303, 0.292, 0.173, 0.292])
    hamacher_tn_upperBound = np.array([0.273, 0.303, 0.292, 0.173, 0.292])
    yager_tn_upperBound_p_0_83 = np.array([0.273, 0.303, 0.292, 0.173, 0.292])
    
    return [
        {
            "name": "owafrs",
            "y" : np.array([1, 1, 0, 1, 0]),

            "sim_matrix": np.array([
            [1.00, 0.54, 0.37, 0.19, 0.10],
            [0.54, 1.00, 0.35, 0.29, 0.20],
            [0.37, 0.35, 1.00, 0.55, 0.73],
            [0.19, 0.29, 0.55, 1.00, 0.74],
            [0.10, 0.20, 0.73, 0.74, 1.00]
        ]),

            "expected": 
            {
                "owa_linear":
                {
                    "reichenbach_lowerBound" : reichenbach_lowerBound,
                    "kd_lowerBound" : kd_lowerBound,
                    "luk_lowerBound" : luk_lowerBound,
                    "goedel_lowerBound" : goedel_lowerBound,
                    "goguen_lowerBound" : goguen_lowerBound,
                    "fodor_lowerBound" : fodor_lowerBound,
                    "rescher_lowerBound" : rescher_lowerBound,
                    "yager_lowerBound" : yager_lowerBound,
                    "weber_lowerBound" : weber_lowerBound,

                    "prod_tn_upperBound" : prod_tn_upperBound,
                    "min_tn_upperBound" : min_tn_upperBound,
                    "prod_tn_upperBound" : prod_tn_upperBound,
                    "min_tn_upperBound" : min_tn_upperBound,
                    "einstein_tn_upperBound" : einstein_tn_upperBound,
                    "luk_tn_upperBound" : luk_tn_upperBound,
                    "drastic_tn_upperBound" : drastic_tn_upperBound,
                    "nilpotent_tn_upperBound" : nilpotent_tn_upperBound,
                    "hamacher_tn_upperBound" : hamacher_tn_upperBound,
                    "yager_tn_upperBound_p_0_83" : yager_tn_upperBound_p_0_83
                }
            }
        }
    ]

