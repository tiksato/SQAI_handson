import time
import numpy as np
from itertools import product, combinations_with_replacement
from numba import njit, uint64, typed, types, prange
from numba.np.ufunc.parallel import get_thread_id, get_num_threads



@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def h1a_prod_jit(h1, cvec, total_dim,
                 Ia_offsets, flat_p, flat_q, flat_phase,
                 trans_offsets, flat_II_ll, flat_JI_ll, flat_II_num,
                 small):

    len_a_strings = len(Ia_offsets) - 1
    sigma = np.zeros(total_dim, dtype=np.complex128)

    #print("nthreads =", get_num_threads())

    for I_a in prange(len_a_strings):
        # Range of transitions for this I_a
        trans_start = Ia_offsets[I_a]
        trans_end = Ia_offsets[I_a + 1]

        #tid = get_thread_id()
        #print('###', tid, I_a)

        # 1.Transition loop
        for t in range(trans_start, trans_end):
            p = flat_p[t]
            q = flat_q[t]
            phase = flat_phase[t]
            fac = phase * h1[p, q]
            
            if np.abs(fac) > small:
                
                # Range of effective beta occupation segments for this transition
                seg_start = trans_offsets[t]
                seg_end = trans_offsets[t + 1]

                # 2. Effective segment loop
                for s in range(seg_start, seg_end):
                    II_ll = flat_II_ll[s]
                    JI_ll = flat_JI_ll[s]
                    n_det = flat_II_num[s]

                    # 3. Explicitly-written vector operation
                    for i_det in range(n_det):
                        sigma[II_ll + i_det] += fac * cvec[JI_ll + i_det]

    return sigma



@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def h2aa_prod_jit(h2_asym, cvec, total_dim,
                  Ia_offsets, flat_p, flat_q, flat_r, flat_s, flat_phase, # per transition
                  trans_offsets, flat_II_ll, flat_JI_ll, flat_II_num, # per beta occ segments
                  small):

    len_a_strings = len(Ia_offsets) - 1
    sigma = np.zeros(total_dim, dtype=np.complex128)

    for I_a in prange(len_a_strings):
        # Range of transitions for this I_a
        trans_start = Ia_offsets[I_a]
        trans_end = Ia_offsets[I_a + 1]

        # 1. Transition loop
        for t in range(trans_start, trans_end):
            p = flat_p[t]
            q = flat_q[t]
            r = flat_r[t]
            s = flat_s[t]
            phase = flat_phase[t]
            # -----------------------------------------------------------------
            # Restrictions $p<q$ and $r<s$ eliminates the factor 1/2
            # <pq|rs>（Coulomb - Exchange) with physisist notation
            fac = phase * h2_asym[p, q, r, s]
            # -----------------------------------------------------------------
            if np.abs(fac) > small:
                # Range of effective beta occupation segments for this transition
                seg_start = trans_offsets[t]
                seg_end = trans_offsets[t + 1]

                # 2. Effective segment loop
                for s_seg in range(seg_start, seg_end):
                    II_ll = flat_II_ll[s_seg]
                    JI_ll = flat_JI_ll[s_seg]
                    n_det = flat_II_num[s_seg]

                    # 3. Explicitly-written vector operation
                    for i_det in range(n_det):
                        sigma[II_ll + i_det] += fac * cvec[JI_ll + i_det]
    return sigma



@njit(parallel=True, fastmath=True, boundscheck=False, cache=False, nogil=True)
def h2ab_prod_jit(
    h2_trans, cvec,
    trans1a_for_pq_offsets, trans1a_for_pq_I, trans1a_for_pq_J, trans1a_for_pq_phase,
    trans1a_for_pq_valid_ll, trans1a_for_pq_valid_num, trans1a_for_pq_valid, 
    h2ab_shuffle, I_b_order_shuffled, 
    trans1b_offsets, trans1b_J, trans1b_p, trans1b_q, trans1b_phase,
    det_ll, det_num, bstr_ll, bstr_num, bstr_to_dist,
    small = 1E-12):

    nthreads = get_num_threads()
    #workload = np.zeros(nthreads, dtype=np.int64)

    len_cvec = len(cvec)
    No = h2_trans.shape[0]
    len_b_dist = det_num.shape[1]
    len_b_strings = len(trans1b_offsets) - 1
    max_num_t1a = np.max(trans1a_for_pq_offsets[1:] - trans1a_for_pq_offsets[:-1])
    max_num_t1a_valid = np.max(trans1a_for_pq_valid_num)


    sigma = np.zeros(len_cvec, dtype=np.complex128)
    Da = np.zeros((len_b_strings, max_num_t1a_valid), dtype=np.complex128)
    Vab_buffers = np.zeros((nthreads, max_num_t1a), dtype=np.complex128)

    for p1q1 in range(No*No):
        p1 = p1q1 // No
        q1 = p1q1 % No

        t1a_begin = trans1a_for_pq_offsets[p1q1]
        num_t1a = trans1a_for_pq_offsets[p1q1+1] - t1a_begin

        if num_t1a == 0:
            continue

        # --- 1. Gather ---
        #Da[:,:] = 0.0 initialization not necessary
        for J_b_dist in range(len_b_dist):
            valid_ll = trans1a_for_pq_valid_ll[p1q1, J_b_dist]
            valid_num = trans1a_for_pq_valid_num[p1q1, J_b_dist]
            if valid_num == 0:
                continue

            bstr_begin = bstr_ll[J_b_dist]

            for i_valid in prange(valid_num):
                i_t1a = trans1a_for_pq_valid[valid_ll + i_valid]
                
                J_a = trans1a_for_pq_J[t1a_begin + i_t1a]
                sign_a = trans1a_for_pq_phase[t1a_begin + i_t1a]
                                
                JJ_begin = det_ll[J_a, J_b_dist]
                for cnt in range(det_num[J_a, J_b_dist]):
                    Da[bstr_begin + cnt, i_valid] = sign_a * cvec[JJ_begin + cnt]
                #if sign_a == 1:
                #    for cnt in range(det_num[J_a, J_b_dist]):
                #        Da[bstr_begin + cnt, i_valid] = cvec[JJ_begin + cnt]
                #else:
                #    for cnt in range(det_num[J_a, J_b_dist]):
                #        Da[bstr_begin + cnt, i_valid] = -cvec[JJ_begin + cnt]

        # --- 2. Contraction & Scatter ---
        if h2ab_shuffle:
            for loop_idx in prange(len_b_strings):
                tid = get_thread_id()
                Vab = Vab_buffers[tid]           
                I_b = I_b_order_shuffled[loop_idx]
                # Below completely the same
            
                I_b_dist = bstr_to_dist[I_b]
                tb_start = trans1b_offsets[I_b]
                tb_end = trans1b_offsets[I_b + 1]
            
                #Vab[:num_t1a] = 0.0 initialization not necessary, see below; Vab[i_t1a] = 0, after being used.
            
                for tb in range(tb_start, tb_end):
                    J_b = trans1b_J[tb]
            
                    J_b_dist = bstr_to_dist[J_b]
                    valid_ll = trans1a_for_pq_valid_ll[p1q1, J_b_dist]
                    valid_num = trans1a_for_pq_valid_num[p1q1, J_b_dist]
                    if valid_num == 0:
                        continue
            
                    beta_factor = trans1b_phase[tb] * h2_trans[p1, q1, trans1b_p[tb], trans1b_q[tb]]
            
                    if np.abs(beta_factor) < small:
                        continue
            
                    #workload[tid] += valid_num
                    for i_valid in range(valid_num):
                        i_t1a = trans1a_for_pq_valid[valid_ll + i_valid]
                        Vab[i_t1a] += Da[J_b, i_valid] * beta_factor
            
                # Scatter to Thread-Local sigma
                base_I_b = I_b - bstr_ll[I_b_dist]
                for i_t1a in range(num_t1a):
                    val = Vab[i_t1a]
                    if val != 0.0:
                        I_a = trans1a_for_pq_I[t1a_begin + i_t1a]
                        if det_num[I_a, I_b_dist] > 0:
                            II = det_ll[I_a, I_b_dist] + base_I_b
                            sigma[II] += Vab[i_t1a]
                    Vab[i_t1a] = 0.0
        else:
            for I_b in prange(len_b_strings):
                tid = get_thread_id()
                Vab = Vab_buffers[tid]
                # Below completely the same
            
                I_b_dist = bstr_to_dist[I_b]
                tb_start = trans1b_offsets[I_b]
                tb_end = trans1b_offsets[I_b + 1]
            
                #Vab[:num_t1a] = 0.0 initialization not necessary, see below; Vab[i_t1a] = 0, after being used.
            
                for tb in range(tb_start, tb_end):
                    J_b = trans1b_J[tb]
            
                    J_b_dist = bstr_to_dist[J_b]
                    valid_ll = trans1a_for_pq_valid_ll[p1q1, J_b_dist]
                    valid_num = trans1a_for_pq_valid_num[p1q1, J_b_dist]
                    if valid_num == 0:
                        continue
            
                    beta_factor = trans1b_phase[tb] * h2_trans[p1, q1, trans1b_p[tb], trans1b_q[tb]]
            
                    if np.abs(beta_factor) < small:
                        continue
            
                    #workload[tid] += valid_num
                    for i_valid in range(valid_num):
                        i_t1a = trans1a_for_pq_valid[valid_ll + i_valid]
                        Vab[i_t1a] += Da[J_b, i_valid] * beta_factor
            
                # Scatter to Thread-Local sigma
                base_I_b = I_b - bstr_ll[I_b_dist]
                for i_t1a in range(num_t1a):
                    val = Vab[i_t1a]
                    if val != 0.0:
                        I_a = trans1a_for_pq_I[t1a_begin + i_t1a]
                        if det_num[I_a, I_b_dist] > 0:
                            II = det_ll[I_a, I_b_dist] + base_I_b
                            sigma[II] += Vab[i_t1a]
                    Vab[i_t1a] = 0.0

    return sigma #, workload



@njit(parallel=True, fastmath=True)
def calc_hdiag_jit(
        h1eff, h2_phys,
        active_a_indices, active_b_indices,
        trans1a_diag_p, trans2a_diag_p, trans2a_diag_q,
        trans1b_diag_p, trans2b_diag_p, trans2b_diag_q):
    """
    Hamiltonian diagonal elements
    """
    n_active = len(active_a_indices)
    hdiag = np.zeros(n_active, dtype=np.complex128)
    
    # Number of electrons and number of pairs
    n_elec_a = trans1a_diag_p.shape[1]
    n_pair_a = trans2a_diag_p.shape[1]
    
    n_elec_b = trans1b_diag_p.shape[1]
    n_pair_b = trans2b_diag_p.shape[1]

    for idx in prange(n_active):
        a_idx = active_a_indices[idx]
        b_idx = active_b_indices[idx]
        
        val = 0.0 + 0.0j
        
        # ==========================================
        # 1. Alpha Self-Energy (1-elec & 2-elec)
        # ==========================================
        for k in range(n_elec_a):
            p = trans1a_diag_p[a_idx, k]
            val += h1eff[p, p]
            
        for m in range(n_pair_a):
            p = trans2a_diag_p[a_idx, m]
            q = trans2a_diag_q[a_idx, m]
            val += h2_phys[p, q, p, q] - h2_phys[p, q, q, p]

        # ==========================================
        # 2. Beta Self-Energy (1-elec & 2-elec)
        # ==========================================
        for k in range(n_elec_b):
            p = trans1b_diag_p[b_idx, k]
            val += h1eff[p, p]
            
        for m in range(n_pair_b):
            p = trans2b_diag_p[b_idx, m]
            q = trans2b_diag_q[b_idx, m]
            val += h2_phys[p, q, p, q] - h2_phys[p, q, q, p]

        # ==========================================
        # 3. Alpha-Beta Interaction (Coulomb only)
        # ==========================================
        for ka in range(n_elec_a):
            p = trans1a_diag_p[a_idx, ka]
            for kb in range(n_elec_b):
                q = trans1b_diag_p[b_idx, kb]
                val += h2_phys[p, q, p, q]

        hdiag[idx] = val

    return hdiag



@njit(parallel=True, fastmath=True, boundscheck=False, cache=False, nogil=True)
def h2ab_prod_jit_opt4(
    h2_trans, cvec,
    trans1a_for_pq_offsets,
    trans1a_for_pq_I, trans1a_for_pq_J, trans1a_for_pq_phase,
    trans1b_offsets,
    trans1b_J, trans1b_p, trans1b_q, trans1b_phase,
    det_ll, det_num, bstr_ll, bstr_num, bstr_to_dist,
    small = 1E-12):

    No = h2_trans.shape[0]
    len_b_dist = det_num.shape[1]
    len_b_strings = len(trans1b_offsets) - 1
    max_num_t1a = np.max(trans1a_for_pq_offsets[1:] - trans1a_for_pq_offsets[:-1])
    len_cvec = len(cvec)

    nthreads = get_num_threads()
    
    Da = np.zeros((len_b_strings, max_num_t1a), dtype=np.complex128)
    Vab_buffers = np.zeros((nthreads, max_num_t1a), dtype=np.complex128)
    sigma = np.zeros(len_cvec, dtype=np.complex128)

    for p1q1 in range(No*No):
        p1 = p1q1 // No
        q1 = p1q1 % No

        t1a_begin = trans1a_for_pq_offsets[p1q1]
        num_t1a = trans1a_for_pq_offsets[p1q1+1] - t1a_begin

        if num_t1a == 0:
            continue
        
        # --- 1. Gather ---
        Da[:,:] = 0.0        
        for i_t1a in prange(num_t1a):
            J_a = trans1a_for_pq_J[t1a_begin + i_t1a]
            sign_a = trans1a_for_pq_phase[t1a_begin + i_t1a]
            for J_b_dist in range(len_b_dist):
                if det_num[J_a, J_b_dist] > 0:
                    bstr_begin = bstr_ll[J_b_dist]
                    JJ_begin = det_ll[J_a, J_b_dist]
                    for cnt in range(det_num[J_a, J_b_dist]):
                        Da[bstr_begin+cnt, i_t1a] = sign_a * cvec[JJ_begin+cnt]
        
        # --- 2. Contraction & Scatter ---
        for I_b in prange(len_b_strings):
            tid = get_thread_id()
            Vab = Vab_buffers[tid]
            
            I_b_dist = bstr_to_dist[I_b]
            tb_start = trans1b_offsets[I_b]
            tb_end = trans1b_offsets[I_b + 1]
        
            Vab[:num_t1a] = 0.0
        
            for tb in range(tb_start, tb_end):
                J_b = trans1b_J[tb]
                
                ############################
                J_b_dist = bstr_to_dist[J_b]
                ############################
                
                beta_factor = trans1b_phase[tb] * h2_trans[p1, q1, trans1b_p[tb], trans1b_q[tb]]
            
                if np.abs(beta_factor) < small:
                    continue
            
                for i_t1a in range(num_t1a):
                    ############################
                    #J_a = trans1a_for_pq_J[t1a_begin + i_t1a]
                    #if det_num[J_a, J_b_dist] == 0:
                    #    continue
                    ############################
                    
                    Vab[i_t1a] += Da[J_b, i_t1a] * beta_factor

            # Scatter to Thread-Local sigma
            for i_t1a in range(num_t1a):
                I_a = trans1a_for_pq_I[t1a_begin + i_t1a]
                if det_num[I_a, I_b_dist] > 0:
                    II = det_ll[I_a, I_b_dist] + I_b - bstr_ll[I_b_dist]
                    sigma[II] += Vab[i_t1a]

    return sigma



@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def h2ab_prod_jit_opt3(
    h2_trans, cvec,
    trans1a_for_pq_offsets,
    trans1a_for_pq_I, trans1a_for_pq_J, trans1a_for_pq_phase,
    trans1b_offsets,
    trans1b_J, trans1b_p, trans1b_q, trans1b_phase,
    det_ll, det_num, bstr_ll, bstr_num, bstr_to_dist,
    small = 1E-12):

    No = h2_trans.shape[0]
    len_b_dist = det_num.shape[1]
    len_b_strings = len(trans1b_offsets) - 1
    max_num_t1a = np.max(trans1a_for_pq_offsets[1:] - trans1a_for_pq_offsets[:-1])
    len_cvec = len(cvec)

    nthreads = get_num_threads()
    
    Da_buffers = np.zeros((nthreads, len_b_strings, max_num_t1a), dtype=np.complex128)
    Vab_buffers = np.zeros((nthreads, max_num_t1a), dtype=np.complex128)
    sigma_buffers = np.zeros((nthreads, len_cvec), dtype=np.complex128)

    for p1q1 in prange(No*No):
        p1 = p1q1 // No
        q1 = p1q1 % No

        t1a_begin = trans1a_for_pq_offsets[p1q1]
        num_t1a = trans1a_for_pq_offsets[p1q1+1] - t1a_begin

        if num_t1a == 0:
            continue

        tid = get_thread_id()
        
        Da = Da_buffers[tid]
        Vab = Vab_buffers[tid]
        sigma_local = sigma_buffers[tid]

        Da[:, :] = 0.0

        # --- 1. Gather ---
        for i_t1a in range(num_t1a):
            J_a = trans1a_for_pq_J[t1a_begin + i_t1a]
            sign_a = trans1a_for_pq_phase[t1a_begin + i_t1a]
            for b_dist in range(len_b_dist):
                if det_num[J_a, b_dist] > 0:
                    bstr_begin = bstr_ll[b_dist]
                    JJ_begin = det_ll[J_a, b_dist]
                    for cnt in range(det_num[J_a, b_dist]):
                        Da[bstr_begin + cnt, i_t1a] += sign_a * cvec[JJ_begin + cnt]

        # --- 2. Contraction & Scatter ---
        for I_b in range(len_b_strings):
            b_dist = bstr_to_dist[I_b]
            tb_start = trans1b_offsets[I_b]
            tb_end = trans1b_offsets[I_b + 1]

            Vab[:num_t1a] = 0.0

            for tb in range(tb_start, tb_end):
                J_b = trans1b_J[tb]
                beta_factor = trans1b_phase[tb] * h2_trans[p1, q1, trans1b_p[tb], trans1b_q[tb]]

                if np.abs(beta_factor) < small:
                    continue

                for i_t1a in range(num_t1a):
                    Vab[i_t1a] += Da[J_b, i_t1a] * beta_factor

            # Scatter to Thread-Local sigma
            for i_t1a in range(num_t1a):
                I_a = trans1a_for_pq_I[t1a_begin + i_t1a]
                if det_num[I_a, b_dist] > 0:
                    II = det_ll[I_a, b_dist] + I_b - bstr_ll[b_dist]
                    sigma_local[II] += Vab[i_t1a]

    sigma = np.sum(sigma_buffers, axis=0)

    return sigma



@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def h2ab_prod_jit_opt2(
    h2_trans,
    cvec,
    trans1a_for_pq_offsets,
    trans1a_for_pq_I,
    trans1a_for_pq_J,
    trans1a_for_pq_phase,
    trans1b_offsets,
    trans1b_J,
    trans1b_p,
    trans1b_q,
    trans1b_phase,
    det_ll,
    det_num,
    bstr_ll,
    bstr_num,
    bstr_to_dist,
    small=1e-12,
):
    # 最終的な出力配列
    sigma = np.zeros(len(cvec), dtype=np.complex128)

    No = h2_trans.shape[0]
    len_b_dist = det_num.shape[1]
    len_b_strings = len(trans1b_offsets) - 1
    max_num_t1a = np.max(
        trans1a_for_pq_offsets[1:] - trans1a_for_pq_offsets[:-1]
    )

    for p1q1 in prange(No * No):
        p1 = p1q1 // No
        q1 = p1q1 % No

        t1a_begin = trans1a_for_pq_offsets[p1q1]
        num_t1a = trans1a_for_pq_offsets[p1q1 + 1] - t1a_begin

        if num_t1a == 0:
            continue

        Da = np.zeros((len_b_strings, max_num_t1a), dtype=np.complex128)
        Vab = np.zeros(max_num_t1a, dtype=np.complex128)
        sigma_local = np.zeros(len(cvec), dtype=np.complex128)

        # --- 1. Gather ---
        for i_t1a in range(num_t1a):
            J_a = trans1a_for_pq_J[t1a_begin + i_t1a]
            sign_a = trans1a_for_pq_phase[t1a_begin + i_t1a]
            for b_dist in range(len_b_dist):
                if det_num[J_a, b_dist] > 0:
                    bstr_begin = bstr_ll[b_dist]
                    JJ_begin = det_ll[J_a, b_dist]
                    for cnt in range(det_num[J_a, b_dist]):
                        Da[bstr_begin + cnt, i_t1a] += (
                            sign_a * cvec[JJ_begin + cnt]
                        )

        # --- 2. Contraction & Scatter ---
        for I_b in range(len_b_strings):
            b_dist = bstr_to_dist[I_b]
            tb_start = trans1b_offsets[I_b]
            tb_end = trans1b_offsets[I_b + 1]

            Vab[:num_t1a] = 0.0

            for tb in range(tb_start, tb_end):
                J_b = trans1b_J[tb]
                beta_factor = (
                    trans1b_phase[tb]
                    * h2_trans[p1, q1, trans1b_p[tb], trans1b_q[tb]]
                )

                if np.abs(beta_factor) < small:
                    continue

                for i_t1a in range(num_t1a):
                    Vab[i_t1a] += Da[J_b, i_t1a] * beta_factor

            # Scatter to LOCAL sigma
            for i_t1a in range(num_t1a):
                I_a = trans1a_for_pq_I[t1a_begin + i_t1a]
                if det_num[I_a, b_dist] > 0:
                    II = det_ll[I_a, b_dist] + I_b - bstr_ll[b_dist]
                    sigma_local[II] += Vab[i_t1a]

        sigma += sigma_local

    return sigma



@njit(parallel=False, fastmath=True, boundscheck=False, cache=True, nogil=True)
def h2ab_prod_jit_opt1(
    h2_trans,
    cvec,
    trans1a_for_pq_offsets,
    trans1a_for_pq_I,
    trans1a_for_pq_J,
    trans1a_for_pq_phase,
    trans1b_offsets,
    trans1b_J,
    trans1b_p,
    trans1b_q,
    trans1b_phase,
    det_ll,
    det_num,
    bstr_ll,
    bstr_num,
    bstr_to_dist,
    small=1e-12,
):
    sigma = np.zeros(len(cvec), dtype=np.complex128)

    No = h2_trans.shape[0]
    len_b_dist = det_num.shape[1]
    len_b_strings = len(trans1b_offsets) - 1
    max_num_t1a = np.max(
        trans1a_for_pq_offsets[1:] - trans1a_for_pq_offsets[:-1]
    )

    Da = np.zeros((max_num_t1a, len_b_strings), dtype=np.complex128)
    Vab = np.zeros(max_num_t1a, dtype=np.complex128)

    for p1q1 in range(No * No):
        p1 = p1q1 // No
        q1 = p1q1 % No

        t1a_begin = trans1a_for_pq_offsets[p1q1]
        num_t1a = trans1a_for_pq_offsets[p1q1 + 1] - t1a_begin

        if num_t1a == 0:
            continue

        # 既存メモリの再利用（スライスクリア）
        Da[:num_t1a, :] = 0.0

        # --- 1. Gather フェーズ ---
        for i_t1a in range(num_t1a):
            J_a = trans1a_for_pq_J[t1a_begin + i_t1a]
            sign_a = trans1a_for_pq_phase[t1a_begin + i_t1a]
            for b_dist in range(len_b_dist):
                if det_num[J_a, b_dist] > 0:
                    bstr_begin = bstr_ll[b_dist]
                    JJ_begin = det_ll[J_a, b_dist]
                    for cnt in range(det_num[J_a, b_dist]):
                        Da[i_t1a, bstr_begin + cnt] += (
                            sign_a * cvec[JJ_begin + cnt]
                        )

        # --- 2. Contraction & Scatter フェーズ ---
        for I_b in range(len_b_strings):
            b_dist = bstr_to_dist[I_b]
            tb_start = trans1b_offsets[I_b]
            tb_end = trans1b_offsets[I_b + 1]

            # 【修正①】新しく生成せず、既存の Vab 領域を 0.0 で上書きする（メモリ確保ゼロ化）
            Vab[:num_t1a] = 0.0

            for tb in range(tb_start, tb_end):
                J_b = trans1b_J[tb]
                beta_factor = (
                    trans1b_phase[tb]
                    * h2_trans[p1, q1, trans1b_p[tb], trans1b_q[tb]]
                )

                if np.abs(beta_factor) < small:
                    continue

                # 【修正②】ループを入れ替え、内側をJ_bにできればベストですが、
                # まずはメモリ非確保の効果を見るためそのままにしています。
                for i_t1a in range(num_t1a):
                    Vab[i_t1a] += Da[i_t1a, J_b] * beta_factor

            # Scatter to sigma
            for i_t1a in range(num_t1a):
                I_a = trans1a_for_pq_I[t1a_begin + i_t1a]
                if det_num[I_a, b_dist] > 0:
                    II = det_ll[I_a, b_dist] + I_b - bstr_ll[b_dist]
                    sigma[II] += Vab[i_t1a]

    return sigma




@njit(parallel=False, fastmath=True, boundscheck=False, cache=True, nogil=True)
def h2ab_prod_jit_opt0(
        h2_trans, cvec, 
        trans1a_for_pq_offsets, 
        trans1a_for_pq_I, trans1a_for_pq_J, trans1a_for_pq_phase, 
        trans1b_offsets, 
        trans1b_J, trans1b_p, trans1b_q, trans1b_phase, 
        det_ll, det_num, bstr_ll, bstr_num, bstr_to_dist, 
        small = 1E-12):

    sigma = np.zeros(len(cvec), dtype=np.complex128)

    No = h2_trans.shape[0]
    len_b_dist = det_num.shape[1]
    len_b_strings = len(trans1b_offsets) - 1
    max_num_t1a = np.max(trans1a_for_pq_offsets[1:]-trans1a_for_pq_offsets[:-1])

    Da = np.zeros((max_num_t1a, len_b_strings), dtype=np.complex128)
    Vab = np.zeros(max_num_t1a, dtype=np.complex128)

    for p1q1 in range(No*No):
        p1 = p1q1 // No
        q1 = p1q1 % No

        t1a_begin = trans1a_for_pq_offsets[p1q1]
        num_t1a = trans1a_for_pq_offsets[p1q1+1] - t1a_begin

        if num_t1a == 0:
            continue

        Da[:num_t1a, :] = 0.0
        for i_t1a in range(num_t1a):            
            J_a = trans1a_for_pq_J[t1a_begin + i_t1a]
            sign_a = trans1a_for_pq_phase[t1a_begin + i_t1a]

            for b_dist in range(len_b_dist):
                if det_num[J_a, b_dist] > 0:
                    bstr_begin = bstr_ll[b_dist]
                    JJ_begin = det_ll[J_a, b_dist]
                    for cnt in range(det_num[J_a, b_dist]):
                        Da[i_t1a, bstr_begin + cnt] \
                            += sign_a * cvec[JJ_begin + cnt]
                        
        for I_b in range(len_b_strings):                            
            b_dist = bstr_to_dist[I_b]
            tb_start = trans1b_offsets[I_b]
            tb_end = trans1b_offsets[I_b + 1]

            Vab[:num_t1a] = np.zeros(num_t1a, dtype=np.complex128)

            for tb in range(tb_start, tb_end):
                J_b = trans1b_J[tb]

                beta_factor = trans1b_phase[tb] \
                    * h2_trans[p1, q1, trans1b_p[tb], trans1b_q[tb]]

                if np.abs(beta_factor) < small:
                    continue

                for i_t1a in range(num_t1a):
                    Vab[i_t1a] += Da[i_t1a, J_b] * beta_factor

            for i_t1a in range(num_t1a):
                I_a = trans1a_for_pq_I[t1a_begin + i_t1a]

                if det_num[I_a, b_dist] > 0:
                    II = det_ll[I_a, b_dist] + I_b - bstr_ll[b_dist]
                    sigma[II] += Vab[i_t1a]

    return sigma



@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def get_determinant_offsets_jit(
        len_a_strings, len_b_dist, a_str2dist, num_b_sorted_strings, mat_allowed
):
 
    det_allowed = np.zeros((len_a_strings, len_b_dist), dtype=np.int8)
    #thread-unsafe: det_ll  = np.zeros((len_a_strings, len_b_dist), dtype=np.int32)
    det_num = np.zeros((len_a_strings, len_b_dist), dtype=np.int32)

    #thread-unsafe: ll_ptr = 0
    for I_a in prange(len_a_strings):
        I_a_dist= a_str2dist[I_a]
        for I_b_dist in range(len_b_dist):
            num_str_b = num_b_sorted_strings[I_b_dist]
            det_allowed[I_a, I_b_dist] = mat_allowed[I_a_dist, I_b_dist]

            #thread-unsafe: det_ll[I_a, I_b_dist] = ll_ptr
            if det_allowed[I_a, I_b_dist] == 1:
                #thread-unsafe: ll_ptr += num_str_b
                det_num[I_a, I_b_dist] = num_str_b

    flat_num = det_num.ravel()
    flat_ll = np.zeros(flat_num.shape, dtype=np.int32)
    flat_ll[1:] = np.cumsum(flat_num)[:-1]
    det_ll = flat_ll.reshape((len_a_strings, len_b_dist))

    return det_allowed, det_ll, det_num



@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def transpose_forward_jit(vec, transpose_map, total_dim):
    """
    C(I_a, I_b) -> C(I_b, I_a)
    """
    vec_T = np.empty(total_dim, dtype=np.complex128)
    for i in prange(total_dim):
        vec_T[transpose_map[i]] = vec[i]
    return vec_T



@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def transpose_backward_jit(vec_T, transpose_map, total_dim):
    """
    C(I_b, I_a) -> C(I_a, I_b)
    """
    vec_orig = np.empty(total_dim, dtype=np.complex128)
    for i in prange(total_dim):
        vec_orig[i] = vec_T[transpose_map[i]]
    return vec_orig



@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def make_transpose_map_jit(
    len_a_strings, len_b_dist, a_str2dist, dist_start_a, dist_start_b,
    det_allowed_a, det_ll_a, det_num_a, det_ll_b, total_dim
):
    """
    make a map for logical transpose of cvec ``matrix''
    """
    transpose_map = np.empty(total_dim, dtype=np.int64)
    
    # 元のベクトルの外側ループ（I_a）を並列走査
    for I_a in prange(len_a_strings):
        I_a_dist = a_str2dist[I_a]
        I_a_local = I_a - dist_start_a[I_a_dist]
        
        for I_b_dist in range(len_b_dist):
            if det_allowed_a[I_a, I_b_dist] == 1:
                ll_orig = det_ll_a[I_a, I_b_dist]
                num_b = det_num_a[I_a, I_b_dist]
                b_start = dist_start_b[I_b_dist]
                
                for i_b_local in range(num_b):
                    I_b = b_start + i_b_local
                    idx_orig = ll_orig + i_b_local
                    
                    # 【一般化の核心】転置先（Beta外側）のパッキング構造 `det_ll_b` を使用
                    idx_trans = det_ll_b[I_b, I_a_dist] + I_a_local
                    
                    transpose_map[idx_orig] = idx_trans
                    
    return transpose_map



@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def get_trans1a_det_boundaries_jit(
        len_a_strings,
        len_b_dist,
        det_allowed,
        det_ll,
        det_num,
        trans1a_counts,
        trans1a_J
):
    '''
    # Offset and number of determinants
    '''
    max_counts = np.max(trans1a_counts)

    II_ll  = np.zeros((len_a_strings, max_counts, len_b_dist), dtype=np.int64)
    II_num = np.zeros((len_a_strings, max_counts, len_b_dist), dtype=np.int64)
    JI_ll  = np.zeros((len_a_strings, max_counts, len_b_dist), dtype=np.int64)
    JI_num = np.zeros((len_a_strings, max_counts, len_b_dist), dtype=np.int64)

    for I_a in prange(len_a_strings):

        for i_trans in range(trans1a_counts[I_a]):
            J_a = trans1a_J[I_a, i_trans]

            for _i in range(len_b_dist):
                both_allowed = det_allowed[I_a, _i] * det_allowed[J_a, _i]
                if both_allowed:
                    II_ll [I_a, i_trans, _i] = det_ll [I_a, _i]
                    II_num[I_a, i_trans, _i] = det_num[I_a, _i]
                    JI_ll [I_a, i_trans, _i] = det_ll [J_a, _i]
                    JI_num[I_a, i_trans, _i] = det_num[J_a, _i]

    return II_ll, II_num, JI_ll, JI_num



@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def transpose_cvec_jit(vec, transpose_map, total_dim):
    vec_T = np.empty(total_dim, dtype=np.complex128)
    for i in prange(total_dim):
        vec_T[transpose_map[i]] = vec[i]
    return vec_T



@njit(parallel=False)
def get_trans1a_for_pq_valid(No, trans1a_for_pq_offsets, trans1a_for_pq_J, det_num):

    len_b_dist = det_num.shape[1]
    max_num_t1a = np.max(trans1a_for_pq_offsets[1:] - trans1a_for_pq_offsets[:-1])

    n_valid = np.zeros((No*No, len_b_dist), dtype=np.int32)
    valid = -1*np.ones((No*No, len_b_dist, max_num_t1a), dtype=np.int32)

    for p1q1 in range(No*No):
        p1 = p1q1 // No
        q1 = p1q1 % No

        t1a_begin = trans1a_for_pq_offsets[p1q1]
        num_t1a = trans1a_for_pq_offsets[p1q1+1] - t1a_begin

        #if num_t1a == 0:
        #    continue

        for i_t1a in range(num_t1a):
            J_a = trans1a_for_pq_J[t1a_begin + i_t1a]
            for J_b_dist in range(len_b_dist):
                if det_num[J_a, J_b_dist] > 0:
                    valid[p1q1, J_b_dist, n_valid[p1q1, J_b_dist]] = i_t1a                    
                    n_valid[p1q1, J_b_dist] += 1

    ll_valid = np.array([0]+list(np.cumsum(n_valid)), 
                       dtype=np.int32)[:-1].reshape(No*No, len_b_dist)

    return ll_valid, n_valid, valid
