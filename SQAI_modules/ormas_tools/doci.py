import time
import math
import numpy as np
from opt_einsum import contract
from itertools import product, combinations_with_replacement
from .ormas1 import ORMAS1
from .doci_jit import *



class DOCI(ORMAS1):
    def __init__(self, n_orb, n_pair, 
                 orb_idx_1 = None,
                 orb_idx_2 = None,
                 num_threads = 1):

        
        occ_info = [{'n_orb':n_orb, 'min':-1, 'max':-1}]
        super().__init__(n_pair, occ_info,
                         initialize_all = False,
                         num_threads = num_threads)

        self.total_dim = self.nstrings

        self.nCr_table = np.zeros((self.n_orb+1, self.n_par+1), dtype=np.int64)
        for n in range(self.n_orb + 1):
            for r in range(self.n_par + 1):
                if n >= r:
                    self.nCr_table[n, r] = math.comb(n, r)


        idx_def = np.array(list(range(self.n_orb)), dtype=np.int8)
        self.orb_idx_1 = orb_idx_1 if orb_idx_1 is not None else idx_def
        self.orb_idx_2 = orb_idx_2 if orb_idx_2 is not None else idx_def



    def get_doci_matrix_elements(self, int1e, int2e):
        return self.get_doci_matrix_elements_general(int1e, int2e)



    def get_doci_matrix_elements_qchem(self, int1e, int2e):
        eps = 2*contract("ii->i", int1e) + contract("iiii->i", int2e)
        g2 = contract("iijj->ij", int2e)
        v2 = 2*contract("ijij->ij", int2e) - contract("ijji->ij", int2e)    
        return eps, g2, v2



    def get_doci_matrix_elements_general(self, int1e, int2e):
        int1a = int1e[np.ix_(self.orb_idx_1, self.orb_idx_1)].copy()
        int1b = int1a[np.ix_(self.orb_idx_2, self.orb_idx_2)].copy()
        int2aa = int2e[np.ix_(self.orb_idx_1, self.orb_idx_1, self.orb_idx_1, self.orb_idx_1)]
        int2bb = int2e[np.ix_(self.orb_idx_2, self.orb_idx_2, self.orb_idx_2, self.orb_idx_2)]    
        int2ab = int2e[np.ix_(self.orb_idx_1, self.orb_idx_2, self.orb_idx_1, self.orb_idx_2)]
        int2ba = int2e[np.ix_(self.orb_idx_2, self.orb_idx_1, self.orb_idx_2, self.orb_idx_1)]

        eps =(contract("ii->i", int1a)
            + contract("ii->i", int1b)
            + 0.5 * (contract("iiii->i", int2ab)
                   + contract("iiii->i", int2ba)))
        
        g2 = 0.5 * (contract("iijj->ij", int2ab)
                  + contract("iijj->ij", int2ba))
        
        v2 = 0.5 * (contract("ijij->ij", int2ab)
                  + contract("ijij->ij", int2ba)
                  + contract("ijij->ij", int2aa) - contract("ijji->ij", int2aa)
                  + contract("ijij->ij", int2bb) - contract("ijji->ij", int2bb))
        return eps, g2, v2


    def calc_hdiag(self, int1e, int2e):

        eps, g2, v2 = self.get_doci_matrix_elements(int1e, int2e)
        h_diag = get_diag_onthefly_64_jit(
            eps,
            v2,
            self.strings_mask
        )

        return h_diag

    def h_prod(self, int1e, int2e, cvec):

        eps, g2, v2 = self.get_doci_matrix_elements(int1e, int2e)
        h_diag = self.calc_hdiag(int1e, int2e)
        sigma = h_prod_onthefly_64_jit_v3(
            h_diag,
            g2,
            cvec,
            self.strings_mask,
            self.n_orb,
            self.n_par,
            self.nCr_table)

        return sigma
    
    
