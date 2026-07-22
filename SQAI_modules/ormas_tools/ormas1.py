import time
import math
import numpy as np
from itertools import product, combinations_with_replacement
from numba import set_num_threads, get_num_threads
from .ormas1_jit import *



class ORMAS1:
    def __init__(self, n_par, occ_info,
                 num_threads = 1,
                 initialize_all = True,
                 verbose = 0):

        self.verbose = verbose
        
        self.num_threads = num_threads
        set_num_threads(self.num_threads)

        if self.verbose > 0:
            print("ORMAS1: num_threads = ", self.num_threads)

        # Parsing
        #############################################
        self.n_par = n_par                 # number of particles
        self.occ_info = occ_info.copy()    # [{'n_orb', 'min', 'max'}]
        self.n_groups = len(self.occ_info) # number of orbital groups
        self.n_orb = 0                     # total number of orbitals
        self.n_orb_sub = []                # number of orbitals in each group
        for info in self.occ_info:
            self.n_orb += info['n_orb']
            self.n_orb_sub.append(info['n_orb'])

        # Generate occupation distributions
        #############################################
        self._generate_occupation_distributions()
        
        # Generate strings for each allowed distribution
        #############################################
        self._generate_strings()
        #self._generate_mapping_arrays()

        if initialize_all:
        
            # Generate string-->index map
            #############################################
            self._generate_string_to_index()

            # Generate one-body transition table
            #############################################
            self._generate_trans1()
            self._generate_trans1_for_pq()
        
            # Generate one-body transition table
            #############################################
            self._generate_trans2()



    def get_mask(self):
        return np.array([s.to_tuple()[0] for s in self.strings])



    def get_fci_indices(self):

        fci = ORMAS1(self.n_par, 
                     occ_info = [{
                         'n_orb':self.n_orb,
                         'min':-1,
                         'max':-1},])
        
        mask_fci = fci.get_mask()
        mask_self = self.get_mask()
        fci_indices = np.searchsorted(mask_fci, mask_self)
        
        return fci_indices
    
    

    def get_fci_group_indices(self):

        fci_indices = self.get_fci_indices()
        fci_nstrings = math.comb(self.n_orb, self.n_par)
        fci_group_indices = np.full(fci_nstrings, -1, dtype=np.int32)
        fci_group_indices[fci_indices] = self.group_indices
        
        return fci_group_indices



    def get_trans_boundaries_for_ormas2(
            self,
            det_allowed, 
            det_ll,
            det_num,
            trans_order):

        if trans_order == 1:
            trans_counts = self.trans1_counts
            trans_J = self.trans1_J
        elif trans_order == 2:
            trans_counts = self.trans2_counts
            trans_J = self.trans2_J
        else:
            raise NotImplementedError(f"trans_order {trans_order} nyi.")

        (trans_II_ll,
         trans_II_num,
         trans_JI_ll,
         trans_JI_num) = \
             get_trans_boundaries_for_ormas2_jit(
                 len(self.strings),
                 det_allowed,
                 det_ll,
                 det_num,
                 trans_counts,
                 trans_J
             )

        max_trans = trans_II_num.shape[1]
        valid_trans_mask = np.arange(max_trans)[None, :] \
            < trans_counts[:, None]
        mask = (trans_II_num > 0)  # (len_a, max_trans, nseg)
        segs_per_trans = mask.sum(axis=2) # (len_a, max_trans)        

        active_segs_flat = segs_per_trans[valid_trans_mask]
        trans_trans_offsets = np.zeros(len(active_segs_flat) + 1,
                                              dtype=np.int64)
        trans_trans_offsets[1:] = np.cumsum(active_segs_flat)

        trans_flat_II_ll = trans_II_ll[mask]
        trans_flat_JI_ll = trans_JI_ll[mask]
        trans_flat_II_num = trans_II_num[mask]
        trans_flat_JI_num = trans_JI_num[mask]

        return (trans_II_ll, trans_II_num,
                trans_JI_ll, trans_JI_num,
                trans_trans_offsets,
                trans_flat_II_ll, trans_flat_II_num, 
                trans_flat_JI_ll, trans_flat_JI_num)



    def _generate_occupation_distributions(self):
        start = time.perf_counter()
        self.occ_dist = []
        for dist in combinations_with_replacement(
                range(self.n_groups), self.n_par):
            
            occ = [0]*self.n_groups
            for i_group in dist:
                occ[i_group] += 1

            allowed = True
            for i_group in range(self.n_groups):
                n_orb_sub = self.occ_info[i_group]['n_orb']
                occ_min = self.occ_info[i_group]['min']
                occ_max = self.occ_info[i_group]['max']
                allowed = allowed and occ[i_group] <= n_orb_sub
                if occ_min >= 0:
                    allowed = allowed and occ[i_group] >= occ_min
                if occ_max >= 0:
                    allowed = allowed and occ[i_group] <= occ_max
            
            if allowed:
                self.occ_dist.append(occ)
        end = time.perf_counter()
        if self.verbose > 0:
            print(f'ORMAS1: Allowed distributions {end-start:.3f} seconds')



    def _generate_strings(self):
        start = time.perf_counter()
        self.strings = []
        self.string_to_dist = []
        #self.string_to_offset = []
        self.strings_per_dist = {}
        self.nstrings_per_dist = []
        
        if self.n_groups == 1:
            
            limits_array = np.array(self.n_orb_sub, dtype = np.int32)
            for i_dist, (n1,) in enumerate(self.occ_dist):
                _strings = generate_strings_1groups_jit(
                    limits_array, n1
                )
                len_strs = len(_strings)
                
                self.strings += _strings
                self.string_to_dist += [i_dist] * len_strs
                #self.string_to_offset += range(len_strs)

                self.strings_per_dist[i_dist] = _strings
                self.nstrings_per_dist.append(len_strs)
                
            self.nstrings = len(self.strings)
            
        elif self.n_groups == 2:
            
            limits_array = np.array(self.n_orb_sub, dtype = np.int32)
            for i_dist, (n1,n2) in enumerate(self.occ_dist):
                _strings = generate_strings_2groups_jit(
                    limits_array, n1, n2
                )
                len_strs = len(_strings)
                
                self.strings += _strings
                self.string_to_dist += [i_dist] * len_strs
                #self.string_to_offset += range(len_strs)

                self.strings_per_dist[i_dist] = _strings
                self.nstrings_per_dist.append(len_strs)
                
            self.nstrings = len(self.strings)
            
        elif self.n_groups == 3:
            
            limits_array = np.array(self.n_orb_sub, dtype = np.int32)
            for i_dist, (n1,n2,n3) in enumerate(self.occ_dist):
                _strings = generate_strings_3groups_jit(
                    limits_array, n1, n2, n3
                )
                len_strs = len(_strings)
                
                self.strings += _strings
                self.string_to_dist += [i_dist] * len_strs
                #self.string_to_offset += range(len_strs)

                self.strings_per_dist[i_dist] = _strings
                self.nstrings_per_dist.append(len_strs)
                
            self.nstrings = len(self.strings)
            
        elif self.n_groups == 4:
            
            limits_array = np.array(self.n_orb_sub, dtype = np.int32)
            for i_dist, (n1,n2,n3,n4) in enumerate(self.occ_dist):
                _strings = generate_strings_4groups_jit(
                    limits_array, n1, n2, n3, n4
                )
                len_strs = len(_strings)
                
                self.strings += _strings
                self.string_to_dist += [i_dist] * len_strs
                #self.string_to_offset += range(len_strs)

                self.strings_per_dist[i_dist] = _strings
                self.nstrings_per_dist.append(len_strs)
                
            self.nstrings = len(self.strings)
            
        else:
            raise NotImplementedError('self.n_groups > 4 nyi.')

        self.strings_mask = self.get_mask()
        self.string_to_dist = np.array(self.string_to_dist, dtype=np.int32)
        self.group_indices = self.string_to_dist
        #self.string_to_offset = np.array(self.string_to_offset, dtype=np.int32)

        self.string_offsets = np.zeros(len(self.occ_dist) + 1, dtype=np.int32)
        self.string_offsets[1:] = np.cumsum(self.nstrings_per_dist)
            
        end = time.perf_counter()
        if self.verbose > 0:
            print(f'ORMAS1: generate strings {end-start:.3f} seconds')



    def _generate_string_to_index(self):
        start = time.perf_counter()
        self.string_to_index = build_string_to_idx(self.strings)
        end = time.perf_counter()
        if self.verbose > 0:
            print(f'ORMAS1: build_string_to_index {end-start:.3f} seconds')



    def _generate_trans1_diag_64(self):
        start = time.perf_counter()
        self.trans1_diag_p = get_one_body_diagonal_transitions_64_jit(
            self.get_mask(),
            self.n_par)
        end = time.perf_counter()
        if self.verbose > 0:
            print(f'ORMAS1: _generate_trans1_diag {end-start:.3f} seconds')

    def _generate_trans1_diag(self):
        start = time.perf_counter()
        self.trans1_diag_p = get_one_body_diagonal_transitions_jit(
            self.strings,
            self.n_par,
            self.n_orb)
        end = time.perf_counter()
        if self.verbose > 0:
            print(f'ORMAS1: _generate_trans1_diag_old {end-start:.3f} seconds')

    def _generate_trans1_64(self):
        start = time.perf_counter()
        self.trans1_diag_p = get_one_body_diagonal_transitions_jit(
            self.strings,
            self.n_par,
            self.n_orb)
        
        (self.trans1_offsets,
         self.trans1_flat_J,
         self.trans1_flat_p,
         self.trans1_flat_q) = get_one_body_transitions_64_jit(
            self.get_mask(),
            self.n_orb)

        end = time.perf_counter()
        if self.verbose > 0:
            print(f'ORMAS1: _generate_trans1_64 {end-start:.3f} seconds')

        
    def _generate_trans1(self):
        start = time.perf_counter()
        self.trans1_diag_p = get_one_body_diagonal_transitions_jit(
            self.strings,
            self.n_par,
            self.n_orb)
        
        (self.trans1_counts,
         self.trans1_J,
         self.trans1_p,
         self.trans1_q,
         self.trans1_phase) = get_one_body_transitions_jit(
            self.strings,
            self.string_to_index,
            self.n_par,
            self.n_orb)

        
        ##### Pack and flatten ######
        self.trans1_offsets = np.zeros(len(self.trans1_counts)+1,
                                       dtype=np.int32)
        self.trans1_offsets[1:] = np.cumsum(self.trans1_counts)
        
        max_trans1 = np.max(self.trans1_counts)
        valid_trans1_mask = np.arange(max_trans1)[None, :] \
            < self.trans1_counts[:, None]

        self.trans1_flat_J = self.trans1_J[valid_trans1_mask]
        self.trans1_flat_p = self.trans1_p[valid_trans1_mask]
        self.trans1_flat_q = self.trans1_q[valid_trans1_mask]
        self.trans1_flat_phase = self.trans1_phase[valid_trans1_mask]
        #############################
        
        end = time.perf_counter()
        if self.verbose > 0:
            print(f'ORMAS1: _generate_trans1 {end-start:.3f} seconds')



    def _generate_trans1_for_pq(self):
        (self.trans1_for_pq_offsets, 
         self.trans1_for_pq_I, 
         self.trans1_for_pq_J, 
         self.trans1_for_pq_phase) = get_trans1_for_pq(
             self.n_orb, 
             self.trans1_offsets,
             self.trans1_flat_J,
             self.trans1_flat_p,
             self.trans1_flat_q,
             self.trans1_flat_phase)



    def _generate_trans2(self):
        start = time.perf_counter()
        (self.trans2_diag_p,
         self.trans2_diag_q) = get_two_body_diagonal_transitions_jit(
            self.strings,
            self.n_par,
            self.n_orb)
        
        (self.trans2_counts,
         self.trans2_J,
         self.trans2_p,
         self.trans2_q,
         self.trans2_r,
         self.trans2_s,
         self.trans2_phase) = get_two_body_transitions_jit(
            self.strings,
            self.string_to_index,
            self.n_par,
            self.n_orb)

        ##### Pack and flatten ######
        self.trans2_offsets = np.zeros(len(self.trans2_counts)+1,
                                       dtype=np.int32)
        self.trans2_offsets[1:] = np.cumsum(self.trans2_counts)
        
        max_trans2 = np.max(self.trans2_counts)
        valid_trans2_mask = np.arange(max_trans2)[None, :] \
            < self.trans2_counts[:, None]

        self.trans2_flat_J = self.trans2_J[valid_trans2_mask]
        self.trans2_flat_p = self.trans2_p[valid_trans2_mask]
        self.trans2_flat_q = self.trans2_q[valid_trans2_mask]
        self.trans2_flat_r = self.trans2_r[valid_trans2_mask]
        self.trans2_flat_s = self.trans2_s[valid_trans2_mask]
        self.trans2_flat_phase = self.trans2_phase[valid_trans2_mask]
        #############################

        end = time.perf_counter()
        if self.verbose > 0:
            print(f'ORMAS1: _generate_trans2 {end-start:.3f} seconds')



