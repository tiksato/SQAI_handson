import time
import numpy as np
from itertools import product, combinations_with_replacement
from numba import njit, uint64, typed, types, prange
from numba.np.ufunc.parallel import get_thread_id, get_num_threads



@njit(fastmath=True, cache=True)
def binary_search_64(arr, target):
    """ソート済みのuint64配列から、対象のストリングインデックス(J)を2分探索"""
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1



@njit(fastmath=True, cache=True)
def get_doci_index(mask, n_orbitals, n_pairs, nCr_table):
    """
    2進数として数値が小さい順（昇順）に並んだ string_array に
    100%完全に同期する O(1) インデックス計算
    """
    idx = 0
    r = n_pairs  # 残りの立っているべきビット（ペア）数

    # 💡 修正：上位ビット（n_orbitals - 1）から下位ビット（0）に向かって走査する
    for p in range(n_orbitals - 1, -1, -1):
        if (mask >> p) & 1:
            # ビットが立っている場合：
            # 昇順ソートの世界では、「このビットが立っていない（0だった）配置」が
            # 自分より前にすべて並んでいることになります。
            # したがって、この位置（p）から残り（r）を選ぶ組合せ数をすべてスキップ（加算）します。
            idx += nCr_table[p, r]
            r -= 1
            if r == 0:
                break

    return idx



@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def get_diag_jit(eps, v2, trans1_diag):

    total_dim = trans1_diag.shape[0]
    n_elec = trans1_diag.shape[1]
    diag_H = np.zeros(total_dim, dtype=np.complex128)

    for I in prange(total_dim):

        diag_E = 0.0
        for i_idx in range(n_elec):
            i = int(trans1_diag[I, i_idx])
            diag_E += eps[i]
            for j_idx in range(n_elec):
                j = int(trans1_diag[I, j_idx])
                if i != j:
                    diag_E += v2[i, j]
        diag_H[I] = diag_E
    return diag_H



@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def get_diag_onthefly_64_jit(eps, v2, string_array):
    """
    On-the-fly version
    """
    n_orbitals = eps.shape[0]
    total_dim = string_array.shape[0]
    diag_H = np.empty(total_dim, dtype=np.complex128)

    for I in prange(total_dim):

        mask = string_array[I]
        diag_E = 0.0
        for p in range(n_orbitals):
            if (mask >> p) & 1:
                diag_E += eps[p]
                for q in range(p + 1, n_orbitals):
                    if (mask >> q) & 1:
                        diag_E += 2.0 * v2[p, q]
        diag_H[I] = diag_E

    return diag_H



@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def h_prod_jit(eps, g2, v2, cvec,
               trans1_diag,
               trans1_offsets,
               trans1_p,
               trans1_q,
               trans1_J,
               small):

    total_dim = len(cvec)
    n_elec = trans1_diag.shape[1]
    sigma = np.zeros(total_dim, dtype=np.complex128)


    for I in prange(total_dim):
        # =================                                                                   
        # 1. Diagonal terms                                                                   
        # =================                                                                   
        diag_E = 0.0
        for i_idx in range(n_elec):
            i = trans1_diag[I, i_idx]
            diag_E += eps[i]  # 2*h_ii + (ii|ii)                                              
            for j_idx in range(n_elec):
                j = trans1_diag[I, j_idx]
                if i != j:
                    # Running all i != j pairs sums both (i,j) and (j,i),                     
                    # thus automatically amounting to 2 * v2[i,j] = 4*J[i,j] - 2K[i,j]        
                    diag_E += v2[i, j]
        sigma[I] += diag_E * cvec[I]
        
        # =====================                                                      
        # 2. Off-diagonal terms                                                      
        # =====================                                                      
        # Range of transitions for this I                                            
        trans_start = trans1_offsets[I]
        trans_end = trans1_offsets[I + 1]

        # Transition loop                                                            
        for t in range(trans_start, trans_end):
            p = trans1_p[t]
            q = trans1_q[t]
            J = trans1_J[t]
            if p == q:
                continue
            sigma[I] += g2[p, q] * cvec[J]

    return sigma

@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def h_prod_onthefly_64_jit(h1_diag, g2, c_vec,
                           string_array, n_orbitals, n_pairs, nCr_table):
    """                                                                                                            
    On-the-fly version                                                                                             
    """
    n_strings = string_array.shape[0]
    sigma_vec = np.empty(n_strings, dtype=np.complex128)

    for I in prange(n_strings):
        mask = string_array[I]
        val = h1_diag[I] * c_vec[I]

        for p in range(n_orbitals):
            if (mask >> p) & 1:
                for q in range(n_orbitals):
                    if p != q and not ((mask >> q) & 1):
                        dest_mask = mask ^ (1 << p) ^ (1 << q)
                        J = get_doci_index(dest_mask, n_orbitals, n_pairs, nCr_table)
                        val += g2[p, q] * c_vec[J]
                #for q in range(p + 1, n_orbitals):
                #    if not ((mask >> q) & 1):
                #        dest_mask = mask ^ (1 << p) ^ (1 << q)
                #        J = get_doci_index(dest_mask, n_orbitals, n_pairs, nCr_table)
                #        val += (g2[p, q] + g2[p, q]) * c_vec[J]
        sigma_vec[I] = val

    return sigma_vec

@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def h_prod_onthefly_64_jit_v2(h1_diag, g2, c_vec,
                              string_array, n_orbitals, n_pairs, nCr_table):
    
    n_strings = string_array.shape[0]
    sigma_vec = np.zeros(n_strings, dtype=np.complex128)

    # identifying occupied and virtual orbitals
    n_occ = n_pairs
    n_vir = n_orbitals - n_pairs

    for I in prange(n_strings):
        mask = string_array[I]
        val = h1_diag[I] * c_vec[I]

        occ_orbs = np.empty(n_occ, dtype=np.int32)
        vir_orbs = np.empty(n_vir, dtype=np.int32)
        
        idx_occ = 0
        idx_vir = 0

        # occupation scan HERE
        for p in range(n_orbitals):
            if (mask >> p) & 1:
                occ_orbs[idx_occ] = p
                idx_occ += 1
            else:
                vir_orbs[idx_vir] = p
                idx_vir += 1

        # Summation loop without `if`s and `bit operation`s
        for i in range(n_occ):
            p = occ_orbs[i]
            for j in range(n_vir):
                q = vir_orbs[j]
                
                # if p != q not needed sice p \in occ and q \in vir
                dest_mask = mask ^ (1 << p) ^ (1 << q)
                J = get_doci_index(dest_mask, n_orbitals, n_pairs, nCr_table)
                val += g2[p, q] * c_vec[J]

        sigma_vec[I] = val

    return sigma_vec



from numba import njit, prange
import numpy as np

@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def h_prod_onthefly_64_jit_v3(h1_diag, g2, c_vec,
                              string_array, n_orbitals, n_pairs, nCr_table):
    
    n_strings = string_array.shape[0]
    sigma_vec = np.zeros(n_strings, dtype=np.complex128)

    n_occ = n_pairs
    n_vir = n_orbitals - n_pairs

    for I in prange(n_strings):
        mask = string_array[I]
        val = h1_diag[I] * c_vec[I]

        occ_orbs = np.empty(n_occ, dtype=np.int32)
        vir_orbs = np.empty(n_vir, dtype=np.int32)
        
        idx_occ = 0
        idx_vir = 0

        # 💡 必須変更：降順（上位ビットから）でパックする！
        # これにより、occ_orbs[i] のランクは常に (n_pairs - i) になります。
        for p in range(n_orbitals - 1, -1, -1):
            if (mask >> p) & 1:
                occ_orbs[idx_occ] = p
                idx_occ += 1
            else:
                vir_orbs[idx_vir] = p
                idx_vir += 1

        for i in range(n_occ):
            p = occ_orbs[i]
            r_p = n_pairs - i  # 消滅する p のランク
            
            for j in range(n_vir):
                q = vir_orbs[j]
                
                # ベースインデックス I から、消滅する p の寄与を引く
                J = I - nCr_table[p, r_p]
                
                if p > q:
                    # ⬇️ 下方遷移: pとqの間にある占有軌道を「上」へずらす
                    m = i + 1
                    while m < n_occ and occ_orbs[m] > q:
                        occ_m = occ_orbs[m]
                        r_m = n_pairs - m
                        # ランクが1上がる
                        J += nCr_table[occ_m, r_m + 1] - nCr_table[occ_m, r_m]
                        m += 1
                    
                    # 生成される q の新しい寄与を足す
                    r_q = n_pairs - m + 1
                    J += nCr_table[q, r_q]
                    
                else:
                    # ⬆️ 上方遷移: pとqの間にある占有軌道を「下」へずらす
                    m = i - 1
                    while m >= 0 and occ_orbs[m] < q:
                        occ_m = occ_orbs[m]
                        r_m = n_pairs - m
                        # ランクが1下がる
                        J += nCr_table[occ_m, r_m - 1] - nCr_table[occ_m, r_m]
                        m -= 1
                    
                    # 生成される q の新しい寄与を足す
                    r_q = n_pairs - m - 1
                    J += nCr_table[q, r_q]

                val += g2[p, q] * c_vec[J]

        sigma_vec[I] = val

    return sigma_vec
