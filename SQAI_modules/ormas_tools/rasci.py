import time
import numpy as np
from itertools import product, combinations_with_replacement
from numba import set_num_threads, get_num_threads
from .ormas2 import ORMAS2



class Full_CI(ORMAS2):
    def __init__(self,
                 n_elec, # (n_elec_a, n_elec_b) with n_elec_a >= n_elec_b
                 n_orb,
                 num_threads = 1,
                 h2ab_shuffle = True
                 ):
        
        super().__init__(
            n_par_components = n_elec,
            occ_info_components = [
                [{'n_orb':n_orb, 'min':-1, 'max':-1}],
                [{'n_orb':n_orb, 'min':-1, 'max':-1}]
            ],
            occ_info_total = [
                {'n_orb':2*n_orb, 'min':-1, 'max':-1}
            ], 
            num_threads = num_threads,
            h2ab_shuffle = h2ab_shuffle)



class RAS_CI(ORMAS2):
    def __init__(self,
                 n_elec, # (n_elec_a, n_elec_b) with n_elec_a >= n_elec_b
                 n_orb_RAS, # (n_orb_RAS1, n_orb_RAS2, n_orb_RAS3)
                 max_hole_RAS1, # maximum number of holes in RAS1
                 max_elec_RAS3, # maximum number of particles in RAS3
                 num_threads = 1,
                 h2ab_shuffle = True
                 ):
        
        max1_a = min(n_elec[0], n_orb_RAS[0])
        min1_a = max1_a - max_hole_RAS1
        min3_a = 0
        max3_a = max_elec_RAS3
        occ_a = [{'n_orb':n_orb_RAS[0], 'min':min1_a, 'max':max1_a},
                 {'n_orb':n_orb_RAS[1], 'min':-1,       'max':-1},
                 {'n_orb':n_orb_RAS[2], 'min':min3_a, 'max':max3_a}]

        max1_b = min(n_elec[1], n_orb_RAS[0])
        min1_b = max1_b - max_hole_RAS1
        min3_b = 0
        max3_b = max_elec_RAS3
        occ_b = [{'n_orb':n_orb_RAS[0], 'min':min1_b, 'max':max1_b},
                 {'n_orb':n_orb_RAS[1], 'min':-1,       'max':-1},
                 {'n_orb':n_orb_RAS[2], 'min':min3_b, 'max':max3_b}]

        max1_tot = min(np.sum(n_elec), 2*n_orb_RAS[0])
        min1_tot = max1_tot - max_hole_RAS1
        min3_tot = 0
        max3_tot = max_elec_RAS3
        occ_tot = [{'n_orb':2*n_orb_RAS[0], 'min':min1_tot, 'max':max1_tot},
                   {'n_orb':2*n_orb_RAS[1], 'min':-1,       'max':-1},
                   {'n_orb':2*n_orb_RAS[2], 'min':min3_tot, 'max':max3_tot}]

        super().__init__(
            n_par_components = n_elec,
            occ_info_components = [occ_a, occ_b],
            occ_info_total = occ_tot,
            num_threads = num_threads,
            h2ab_shuffle = h2ab_shuffle)
            


class RHF_CI(RAS_CI):
    def __init__(self,
                 n_elec, # (n_elec_a, n_elec_b) with n_elec_a >= n_elec_b
                 n_orb,
                 max_rank,
                 num_threads = 1,
                 h2ab_shuffle = True
                 ):
        
        super().__init__(
            n_elec = n_elec,
            n_orb_RAS = (n_elec[0], 0, n_orb-n_elec[0]),
            max_hole_RAS1 = max_rank,
            max_elec_RAS3 = max_rank,
            num_threads = num_threads,
            h2ab_shuffle = h2ab_shuffle)
