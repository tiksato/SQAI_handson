import time
import numpy as np
from itertools import product, combinations_with_replacement
from numba import set_num_threads, get_num_threads
from .ormas1 import ORMAS1
from .ormas2_jit import *



class ORMAS2:
    def __init__(self,
                 n_par_components,
                 occ_info_components,
                 occ_info_total,
                 num_threads = 1,
                 h2ab_shuffle = True,
                 verbose = 0
                 ):

        self.verbose = verbose
        
        self.use_jit = True
        self.num_threads = num_threads
        set_num_threads(self.num_threads)
        self.h2ab_shuffle = h2ab_shuffle
        
        self.n_par_components = n_par_components
        self.occ_info_components = occ_info_components
        self.occ_info_total = occ_info_total
        
        self.n_groups = len(self.occ_info_total)
        self.n_components = len(self.n_par_components)
        self.n_par_total = np.sum(self.n_par_components)



        if self.n_components != 2:
            raise Exception('n_components = 2 only.')



        self._generate_ormas1s()        
        self._generate_distributions()
        self._generate_occupation_boundaries()
        self._generate_determinant_offsets()        
        self._generate_transpose_map()
        self._generate_trans1_det_boundaries()
        self._generate_trans2_det_boundaries()
        self._generate_trans1a_for_pq_valid()
        self._generate_packed_indices()

        np.random.seed(42)
        is_full_ci = (self.mat_allowed.shape[1] == 1)
        self.h2ab_shuffle = h2ab_shuffle and (not is_full_ci)
        self.I_b_order_shuffled = np.random.permutation(self.ormas1[1].nstrings).astype(np.int32)

    

    def transpose_forward(self, cvec):
        return transpose_forward_jit(cvec, self.transpose_map, self.total_dim)



    def transpose_backward(self, cvec_T):
        return transpose_backward_jit(cvec_T, self.transpose_map, self.total_dim)



    def generate_idx_maps(self):
        
        bstr_num = self.ormas1[0].nstrings_per_dist
        bstr_offs = np.zeros(len(bstr_num)+1, dtype=np.int32)
        bstr_offs[1:] = np.cumsum(bstr_num)
        bstr_ll = bstr_offs[:-1]


        len_a_strings = self.det_num.shape[0]
        len_b_dist = self.det_num.shape[1]
    
        maska_array = self.ormas1[0].get_mask()
        maskb_array = self.ormas1[1].get_mask()

        mapf = []
        mapr = {}

    
        II = 0
        for I_a in range(len_a_strings):
            maska = maska_array[I_a]
            for I_b_dist in range(len_b_dist):
                if self.det_num[I_a, I_b_dist] > 0:
                    bstr_begin = bstr_ll[I_b_dist]
                    II_begin = self.det_ll[I_a, I_b_dist]
                    for cnt in range(self.det_num[I_a, I_b_dist]):
                        maskb = maskb_array[bstr_begin + cnt]
                        mapf.append((maska, maskb))
                        mapr[(maska, maskb)] = II
                        II += 1

        return mapf, mapr


    
    def h1a_prod(self, h1, cvec, small=1E-12):

        return h1a_prod_jit(
            h1, cvec, self.total_dim,
            self.ormas1[0].trans1_offsets,
            self.ormas1[0].trans1_flat_p,
            self.ormas1[0].trans1_flat_q,
            self.ormas1[0].trans1_flat_phase,
            self.trans1a_trans_offsets,
            self.trans1a_flat_II_ll,
            self.trans1a_flat_JI_ll,
            self.trans1a_flat_II_num,
            small)



    def h1b_prod(self, h1, cvec, small=1E-12):

        cvec_T = transpose_forward_jit(
            cvec, self.transpose_map, self.total_dim
        )
        sigma_T = h1a_prod_jit(
            h1, cvec_T, self.total_dim,
            self.ormas1[1].trans1_offsets,
            self.ormas1[1].trans1_flat_p,
            self.ormas1[1].trans1_flat_q,
            self.ormas1[1].trans1_flat_phase,
            self.trans1b_trans_offsets,
            self.trans1b_flat_II_ll,
            self.trans1b_flat_JI_ll,
            self.trans1b_flat_II_num,
            small)
        return transpose_backward_jit(
            sigma_T, self.transpose_map, self.total_dim
        )


        
    def h2aa_prod(self, h2, cvec, small=1E-12):

        h2_asym = np.ascontiguousarray(h2 - h2.transpose(0,1,3,2))
        return h2aa_prod_jit(
            h2_asym, cvec, self.total_dim,
            self.ormas1[0].trans2_offsets,
            self.ormas1[0].trans2_flat_p,
            self.ormas1[0].trans2_flat_q,
            self.ormas1[0].trans2_flat_r,
            self.ormas1[0].trans2_flat_s,
            self.ormas1[0].trans2_flat_phase,
            self.trans2a_trans_offsets,
            self.trans2a_flat_II_ll,
            self.trans2a_flat_JI_ll,
            self.trans2a_flat_II_num,
            small)



    def h2bb_prod(self, h2, cvec, small=1E-12):

        h2_asym = np.ascontiguousarray(h2 - h2.transpose(0,1,3,2))
        cvec_T = transpose_forward_jit(cvec, self.transpose_map, self.total_dim)
        sigma_T = h2aa_prod_jit(
            h2_asym, cvec_T, self.total_dim,
            self.ormas1[1].trans2_offsets,
            self.ormas1[1].trans2_flat_p,
            self.ormas1[1].trans2_flat_q,
            self.ormas1[1].trans2_flat_r,
            self.ormas1[1].trans2_flat_s,
            self.ormas1[1].trans2_flat_phase,
            self.trans2b_trans_offsets,
            self.trans2b_flat_II_ll,
            self.trans2b_flat_JI_ll,
            self.trans2b_flat_II_num,
            small)
        return transpose_backward_jit(
            sigma_T, self.transpose_map, self.total_dim
        )



    def h2ab_prod(self, h2, cvec, small=1E-12):

        h2_trans = np.ascontiguousarray(h2.transpose(0, 2, 1, 3))

        return h2ab_prod_jit(
            h2_trans, cvec,
            self.ormas1[0].trans1_for_pq_offsets,
            self.ormas1[0].trans1_for_pq_I,
            self.ormas1[0].trans1_for_pq_J,
            self.ormas1[0].trans1_for_pq_phase,
            self.trans1a_for_pq_valid_ll,
            self.trans1a_for_pq_valid_num,
            self.trans1a_for_pq_valid,
            self.h2ab_shuffle, 
            self.I_b_order_shuffled,
            self.ormas1[1].trans1_offsets,
            self.ormas1[1].trans1_flat_J,
            self.ormas1[1].trans1_flat_p,
            self.ormas1[1].trans1_flat_q,
            self.ormas1[1].trans1_flat_phase,
            self.det_ll, self.det_num,
            self.ormas1[1].string_offsets,
            self.ormas1[1].nstrings_per_dist,
            self.ormas1[1].string_to_dist,
            small)



    def calc_hdiag(self, h1eff, h2_phys):
        return calc_hdiag_jit(
            h1eff, h2_phys, 
            self.I_to_Ia, self.I_to_Ib,
            self.ormas1[0].trans1_diag_p, 
            self.ormas1[0].trans2_diag_p, 
            self.ormas1[0].trans2_diag_q, 
            self.ormas1[1].trans1_diag_p, 
            self.ormas1[1].trans2_diag_p, 
            self.ormas1[1].trans2_diag_q)            



    def h_prod(self, h1eff, h2_phys, cvec, small=1E-12):
        sigma = self.h1a_prod(h1eff, cvec, small)
        sigma += self.h1b_prod(h1eff, cvec, small)
        sigma += self.h2aa_prod(h2_phys, cvec, small)
        sigma += self.h2bb_prod(h2_phys, cvec, small)
        sigma += self.h2ab_prod(h2_phys, cvec, small)
        return sigma



    def h_prod_symmetric(self, h1eff, h2_phys, cvec, small=1E-12):
        sigma = self.h1a_prod(h1eff, cvec)
        sigma += self.h2aa_prod(h2_phys, cvec)
        sigma += self.transpose_forward(sigma)
        sigma += self.h2ab_prod(h2_phys, cvec)
        return sigma


    def h_prod_force_symmetric(self, h1eff, h2_phys, cvec, small=1E-12):
        cvec_sym = 0.5*(cvec + self.transpose_forward(cvec))
        sigma = self.h1a_prod(h1eff, cvec_sym)
        sigma += self.h2aa_prod(h2_phys, cvec_sym)
        sigma += self.transpose_forward(sigma)
        sigma += self.h2ab_prod(h2_phys, cvec_sym)
        sigma = 0.5*(sigma + self.transpose_forward(sigma))
        return sigma

        

    def get_fci_mask(self):
        """
        Obtain a 2D boolean mask for Full CI matrix C(I,J)
        """
        fci_group_idx_a = self.ormas1[0].get_fci_group_indices()
        fci_group_idx_b = self.ormas1[1].get_fci_group_indices()

        ng_a, ng_b = self.mat_num_str.shape
        allowed_groups_padded = np.zeros((ng_a + 1, ng_b + 1), dtype=np.bool_)
        allowed_groups_padded[:ng_a, :ng_b] = (self.mat_num_str > 0)

        fci_mask = allowed_groups_padded[fci_group_idx_a[:, None],
                                         fci_group_idx_b[None, :]]
        return fci_mask



    def get_det_mask(self, _check_ = False):
        """
        Generates a 2D boolean mask for FCI coefficients ``matrix''.
        Inputs:
            self.mat_num_str (ndarray)
            str_group_idx_a (list or ndarray)
            str_group_idx_b (list or ndarray)
        Returns:
            mask (ndarray): 2D boolean array of shape (n_str_a, n_str_b).
                            True indicates allowed within the RAS subspace.
        """

        str_group_idx_a = self.ormas1[0].string_to_dist
        str_group_idx_b = self.ormas1[1].string_to_dist
        
        # 1. Convert input indexes into numpy arrays for fast vectorized broadcasting
        group_a = np.array(str_group_idx_a, dtype=np.int32)
        group_b = np.array(str_group_idx_b, dtype=np.int32)
        
        # 2. Identify allowed group combinations from your self.mat_num_str matrix
        #    True if the group block contains configurations, False if it is empty (0)
        allowed_groups = (self.mat_num_str > 0)
        
        # 3. Vectorized broadcasting (Outer Mapping)
        #    group_a[:, None] creates a column vector of shape (n_str_a, 1)
        #    group_b[None, :] creates a row vector of shape (1, n_str_b)
        #    Indexing 'allowed_groups' with these grids creates the 2D FCI-shaped mask.
        mask = allowed_groups[group_a[:, None], group_b[None, :]]
        
        # 4. Defensive Sanity Check: Verify if the masked dimension matches your expectation
        if _check_:
            calculated_dim = np.sum(self.mat_num_str)
            actual_mask_dim = np.count_nonzero(mask)
        
            print(f"\n[Mask Generator] Processing group matrix...")
            print(f"  --> Total Allowed Configurations (sum of matrix): {calculated_dim:,}")
            print(f"  --> Generated 2D Mask Active Elements:            {actual_mask_dim:,}")
            if calculated_dim != actual_mask_dim:
                print("  --> WARNING: Dimension mismatch!")
            else:
                print("  --> Success: Mask generated and validated.")
            
        return mask

    def _generate_ormas1s(self):
        start = time.perf_counter()
        self.ormas1 = [
            ORMAS1(self.n_par_components[i],
                   self.occ_info_components[i],
                   num_threads = self.num_threads,
                   initialize_all = True, 
                   verbose = self.verbose)
            for i in range(self.n_components)
        ]

        # convenient aliases
        self.a_dist = self.ormas1[0].occ_dist
        self.b_dist = self.ormas1[1].occ_dist
        self.a_strings = self.ormas1[0].strings
        self.b_strings = self.ormas1[1].strings

        self.a_dist_start = self.ormas1[0].string_offsets
        self.b_dist_start = self.ormas1[1].string_offsets
        
        self.a_str2dist = self.ormas1[0].string_to_dist
        self.b_str2dist = self.ormas1[1].string_to_dist

        self.a_sorted_strings = self.ormas1[0].strings_per_dist
        self.b_sorted_strings = self.ormas1[1].strings_per_dist
        
        end = time.perf_counter()
        if self.verbose > 0:
            print(f'ORMAS1 objects construction {end-start:.3f} seconds')



    def _generate_distributions(self):        
        start = time.perf_counter()
        occ_dist_components = [
            self.ormas1[i].occ_dist
            for i in range(self.n_components)
        ]        
        self.allowed_distributions = []
        tmp = [range(len(o)) for o in occ_dist_components]
        for iocc, occ in zip(
                product(*tmp), 
                product(*occ_dist_components)
                ):    
            occ_tot = np.sum(np.array(occ), axis=0)
            #print(iocc, occ, occ_tot)

            allowed = True
            for i_group in range(self.n_groups):
                occ_min = self.occ_info_total[i_group]['min']
                if occ_min >= 0:
                    allowed = allowed and occ_tot[i_group] >= occ_min
                occ_max = self.occ_info_total[i_group]['max']
                if occ_max >= 0:
                    allowed = allowed and occ_tot[i_group] <= occ_max        
            #print(iocc, occ, occ_tot, allowed)
            if allowed:
                self.allowed_distributions.append((iocc, occ, occ_tot))
        #len_dist_components = [len(self.ormas1[k].occ_dist)
        #                       for k in range(self.n_components)]
        end = time.perf_counter()
        if self.verbose > 0:
            print(f'generate_distributions {end-start:.3f} seconds')


        
    def _generate_occupation_boundaries(self):
        start = time.perf_counter()
        
        a_dist = self.a_dist
        b_dist = self.b_dist
        a_strings = self.a_strings
        b_strings = self.b_strings
        a_str2dist = self.a_str2dist
        b_str2dist = self.b_str2dist
        a_sorted_strings = self.a_sorted_strings
        b_sorted_strings = self.b_sorted_strings

        
        # Allowed distribution matrix
        self.mat_allowed = np.zeros((len(a_dist), len(b_dist)), dtype=int)
        self.mat_num_str = np.zeros((len(a_dist), len(b_dist)), dtype=int)
        for info in self.allowed_distributions:
            iD_a, iD_b = info[0]
            num_str_a = len(a_sorted_strings[iD_a])
            num_str_b = len(b_sorted_strings[iD_b])
            
            self.mat_allowed[iD_a, iD_b] = 1
            self.mat_num_str[iD_a, iD_b] = num_str_a*num_str_b

        #self.mat_str_ul = np.zeros((len(a_dist), len(b_dist)), dtype=int)
        #for i_a_dist in range(len(self.ormas1[0].occ_dist)):
        #    for i_b_dist in range(len(self.ormas1[1].occ_dist)):
        #        self.mat_str_ul[i_a_dist:, i_b_dist:] += self.mat_num_str[i_a_dist, i_b_dist]
        #self.mat_str_ll = self.mat_str_ul - self.mat_num_str


        # Find possible Dist_b valid for |Dist_a Dist_b>
        valid_Db_for_Da = []
        for iD_a, DistI_a in enumerate(self.ormas1[0].occ_dist):
            valid_Db_for_Da.append(list(np.where(self.mat_allowed[iD_a,:]==1)[0]))

        # Dimension of the full problem
        self.total_dim = 0
        for dist_a in range(len(self.ormas1[0].occ_dist)):    
            num_str_a = len(a_sorted_strings[dist_a])
            for dist_b in valid_Db_for_Da[dist_a]:
                num_str_b = len(b_sorted_strings[dist_b])
                self.total_dim += num_str_a*num_str_b
                #print(dist_a, dist_b, num_str_a, num_str_b, num_str_a*num_str_b, self.total_dim)

        end = time.perf_counter()
        if self.verbose > 0:
            print(f'generate_occupation_boundaries {end-start:.3f} seconds')



    def _generate_determinant_offsets(self):
        start = time.perf_counter()
        
        self.det_allowed, self.det_ll, self.det_num \
            = get_determinant_offsets_jit(
                len(self.a_strings),
                len(self.b_dist),
                self.a_str2dist,
                self.ormas1[1].nstrings_per_dist,
                self.mat_allowed)
        self.det_allowed_b, self.det_ll_b, self.det_num_b \
            = get_determinant_offsets_jit(
                len(self.b_strings),
                len(self.a_dist),
                self.b_str2dist,
                self.ormas1[0].nstrings_per_dist,
                self.mat_allowed.T)
        end = time.perf_counter()
        if self.verbose > 0:
            print(f'generate_determinant_offsets {end-start:.3f} seconds')



    def _generate_transpose_map(self):
        start = time.perf_counter()        
        self.transpose_map = make_transpose_map_jit(
            len(self.a_strings),
            len(self.b_dist),
            self.a_str2dist,
            self.b_dist_start, #dist_start,
            self.a_dist_start, #dist_start_b,
            self.det_allowed,
            self.det_ll,
            self.det_num,
            #self.det_allowed_b,
            self.det_ll_b,
            #self.det_num_b,
            self.total_dim)
        end = time.perf_counter()
        if self.verbose > 0:
            print(f'generate_transpose_map {end-start:.3f} seconds')

    def _generate_trans1_det_boundaries(self):
        start = time.perf_counter()

        (self.trans1a_II_ll, self.trans1a_II_num,
         self.trans1a_JI_ll, self.trans1a_JI_num,
         self.trans1a_trans_offsets,
         self.trans1a_flat_II_ll, self.trans1a_flat_II_num,
         self.trans1a_flat_JI_ll, self.trans1a_flat_JI_num) = \
             self.ormas1[0].get_trans_boundaries_for_ormas2(
                 self.det_allowed,
                 self.det_ll,
                 self.det_num,
                 trans_order = 1)

        (self.trans1b_II_ll, self.trans1b_II_num,
         self.trans1b_JI_ll, self.trans1b_JI_num,
         self.trans1b_trans_offsets,
         self.trans1b_flat_II_ll, self.trans1b_flat_II_num,
         self.trans1b_flat_JI_ll, self.trans1b_flat_JI_num) = \
             self.ormas1[1].get_trans_boundaries_for_ormas2(
                 self.det_allowed_b,
                 self.det_ll_b,
                 self.det_num_b,
                 trans_order = 1)
        
        end = time.perf_counter()
        if self.verbose > 0:
            print(f'generate_trans1_det_boundaries {end-start:.3f} seconds')


        
    def _generate_trans2_det_boundaries(self):
        start = time.perf_counter()

        (self.trans2a_II_ll, self.trans2a_II_num,
         self.trans2a_JI_ll, self.trans2a_JI_num,
         self.trans2a_trans_offsets,
         self.trans2a_flat_II_ll, self.trans2a_flat_II_num,
         self.trans2a_flat_JI_ll, self.trans2a_flat_JI_num) = \
             self.ormas1[0].get_trans_boundaries_for_ormas2(
                 self.det_allowed,
                 self.det_ll,
                 self.det_num,
                 trans_order = 2)

        (self.trans2b_II_ll, self.trans2b_II_num,
         self.trans2b_JI_ll, self.trans2b_JI_num,
         self.trans2b_trans_offsets,
         self.trans2b_flat_II_ll, self.trans2b_flat_II_num,
         self.trans2b_flat_JI_ll, self.trans2b_flat_JI_num) = \
             self.ormas1[1].get_trans_boundaries_for_ormas2(
                 self.det_allowed_b,
                 self.det_ll_b,
                 self.det_num_b,
                 trans_order = 2)

        end = time.perf_counter()
        if self.verbose > 0:
            print(f'generate_trans2_det_boundaries {end-start:.3f} seconds')



    def _generate_trans1a_for_pq_valid(self):
        start = time.perf_counter()
        (self.trans1a_for_pq_valid_ll,
         self.trans1a_for_pq_valid_num,
         valid) = get_trans1a_for_pq_valid(
            self.ormas1[0].n_orb, 
            self.ormas1[0].trans1_for_pq_offsets,
            self.ormas1[0].trans1_for_pq_J,
            self.det_num)

        valid = valid.ravel()
        mask = (valid >= 0)
        self.trans1a_for_pq_valid = valid[mask]

        end = time.perf_counter()
        if self.verbose > 0:
            print(f'generate_trans1_for_pq_valid {end-start:.3f} seconds')



    def _generate_packed_indices(self, to_fci_order=True):
        """
        Generates CI index --> alpha_string index, beta_string index table
        """
        # 1. string lookup arrays
        group_a = self.ormas1[0].string_to_dist
        group_b = self.ormas1[1].string_to_dist
        
        num_groups_a, num_groups_b = self.mat_num_str.shape
        
        # 2. Pre-group string indices by their distribution group (O(N) lookup maps)
        #    a_in_group[g] stores an array of all alpha string indices belonging to group g
        a_in_group = [np.where(group_a == g)[0] for g in range(num_groups_a)]
        b_in_group = [np.where(group_b == g)[0] for g in range(num_groups_b)]
        
        active_a_list = []
        active_b_list = []
        
        # 3. Loop over allowed group blocks
        for g_a in range(num_groups_a):
            for g_b in range(num_groups_b):
                if self.mat_num_str[g_a, g_b] > 0:
                    a_strs = a_in_group[g_a]
                    b_strs = b_in_group[g_b]
                    
                    # Generate all combinations within this specific block using fast CPU vectorization
                    # a_grid repeats each alpha string, b_grid tiles the entire beta string array
                    a_grid = np.repeat(a_strs, len(b_strs))
                    b_grid = np.tile(b_strs, len(a_strs))
                    
                    active_a_list.append(a_grid)
                    active_b_list.append(b_grid)
                    
        # Combine all block segments into the final 1D packed arrays
        active_a_indices = np.concatenate(active_a_list)
        active_b_indices = np.concatenate(active_b_list)
        
        # 4. Order Mapping to replicate np.where(mask) behavior if required
        if to_fci_order:
            # lexsort sorts by the last key first. 
            # This sorts primarily by alpha string (ascending), secondarily by beta string (ascending).
            # This guarantees 100% identity with the layout produced by the 2D mask method.
            sort_idx = np.lexsort((active_b_indices, active_a_indices))
            active_a_indices = active_a_indices[sort_idx]
            active_b_indices = active_b_indices[sort_idx]
            
        self.I_to_Ia = active_a_indices
        self.I_to_Ib = active_b_indices



    def _h2ab_prod_test(self, h2, cvec, small=1E-12, test_type = 0):
        '''
        same as h2ab_prod
        '''
        h2_trans = np.ascontiguousarray(h2.transpose(0, 2, 1, 3))

        if test_type == 0:
            h2ab_prod_test = h2ab_prod_jit_opt0
        elif test_type == 1:
            h2ab_prod_test = h2ab_prod_jit_opt1
        elif test_type == 2:
            h2ab_prod_test = h2ab_prod_jit_opt2
        elif test_type == 3:
            h2ab_prod_test = h2ab_prod_jit_opt3
        elif test_type == 4:
            h2ab_prod_test = h2ab_prod_jit_opt4
        else:
            raise Exception("Bad test_type")

        return h2ab_prod_test(
            h2_trans, cvec,
            self.ormas1[0].trans1_for_pq_offsets,
            self.ormas1[0].trans1_for_pq_I,
            self.ormas1[0].trans1_for_pq_J,
            self.ormas1[0].trans1_for_pq_phase,
            self.ormas1[1].trans1_offsets,
            self.ormas1[1].trans1_flat_J,
            self.ormas1[1].trans1_flat_p,
            self.ormas1[1].trans1_flat_q,
            self.ormas1[1].trans1_flat_phase,
            self.det_ll, self.det_num,
            self.ormas1[1].string_offsets,
            self.ormas1[1].nstrings_per_dist,
            self.ormas1[1].string_to_dist,
            small)
