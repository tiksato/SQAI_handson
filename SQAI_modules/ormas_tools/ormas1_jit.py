import time
import numpy as np
from numba import njit, uint64, typed, types, prange
from numba.np.ufunc.parallel import get_thread_id, get_num_threads
from .bitstring128 import BitString128



@njit
def get_combinations_64(n_orbs, n_electrons):
    """
    Generates all integers with 'n_electrons' bits set within 'n_orbs'.
    Gosper's Hack for fast bit combination generation.
    """
    res = typed.List.empty_list(uint64)
    if n_electrons == 0:
        res.append(uint64(0))
        return res
    if n_electrons > n_orbs:
        return res

    # Initial lexicographically smallest combination
    current = (uint64(1) << uint64(n_electrons)) - uint64(1)
    limit = uint64(1) << uint64(n_orbs)

    while current < limit:
        res.append(current)
        # Gosper's Hack to find the next lexicographically larger combination
        lowbit = current & -current
        tmp = current + lowbit
        current = (((current ^ tmp) >> 2) // lowbit) | tmp
        
        # Safety break for overflow
        if current == 0: break
        
    return res



@njit
def _apply_shifted_mask(s, mask, offset):
    """Helper to apply a 64-bit mask to BitString128 at a specific offset."""
    if offset < 64:
        s.low |= (mask << uint64(offset))
        # Handle overflow into 'high' if mask crosses the 64-bit boundary
        remaining_shift = 64 - offset
        if remaining_shift < 64:
            s.high |= (mask >> uint64(remaining_shift))
    else:
        s.high |= (mask << uint64(offset - 64))
    return s



@njit
def build_string_to_idx(string_list):
    """
    Build a reverse lookup dictionary.
    No explicit type specification to avoid TypingErrors.
    Type is inferred from the first assignment.
    """
    # 1. Create a truly empty dictionary without any type hints    
    s2i = typed.Dict()

    if len(string_list) == 0:
        return s2i
    
    # 2. Iterate and assign
    # The first assignment s2i[key] = value will lock the types:
    # Key: Tuple(uint64, uint64)
    # Value: int32
    for i in range(len(string_list)):
         #s2i[string_list[i]] = types.int32(i)
         s2i[(string_list[i].low, string_list[i].high)] = types.int32(i)

    return s2i



@njit
def get_trans1_for_pq(No, I_offsets, trans1_J, trans1_p, trans1_q, trans1_phase):
        
    len_strings = len(I_offsets) - 1

    _num = np.zeros((No*No), dtype=np.int32)
    for I in range(len_strings):
        for t in range(I_offsets[I], I_offsets[I + 1]):
            _num[trans1_p[t] * No + trans1_q[t]] += 1

    trans1_for_pq_offsets = np.zeros(len(_num) + 1, dtype=np.int64)
    trans1_for_pq_offsets[1:] = np.cumsum(_num)
    _offsets = trans1_for_pq_offsets

    num_t1 = I_offsets[-1]
    trans1_for_pq_I = np.zeros(num_t1, dtype=np.int32)
    trans1_for_pq_J = np.zeros(num_t1, dtype=np.int32)
    trans1_for_pq_phase = np.zeros(num_t1, dtype=np.int8)

    _num = np.zeros((No*No), dtype=np.int32)
    for I in range(len_strings):
        for t in range(I_offsets[I], I_offsets[I + 1]):
            pq = trans1_p[t] * No + trans1_q[t]            
            trans1_for_pq_I[_offsets[pq] + _num[pq]] = I
            trans1_for_pq_J[_offsets[pq] + _num[pq]] = trans1_J[t]
            trans1_for_pq_phase[_offsets[pq] + _num[pq]] = trans1_phase[t]
            _num[pq] += 1

    return (trans1_for_pq_offsets, 
            trans1_for_pq_I, 
            trans1_for_pq_J, 
            trans1_for_pq_phase)



@njit(inline='always')
def _ctz_and_clear_64(mask):
    """
    return the index of the lowest (1) bit and the next mask having this bit anihirated
    """
    # extract the lowest bit by mask & -mask (e.g, 0b00100 -> 4)
    lowest_bit = mask & -mask
    idx = 0
    temp = lowest_bit
    while temp > 1:
        temp >>= 1
        idx += 1
        
    next_mask = mask ^ lowest_bit
    return idx, next_mask


@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)                     
def get_one_body_diagonal_transitions_64_jit(string_array, n_el):
    """
    Args:
        string_array (ndarray): uint64 bitmask array (shape=(n_strings,))
        n_el (int): number of electrons
        
    Returns:
        trans1_diag (ndarray): int32 (n_strings, n_el)
    """
    n_strings = string_array.shape[0]
    trans1_diag = np.empty((n_strings, n_el), dtype=np.int32)

    for I in prange(n_strings):
        mask_val = string_array[I]
        for count in range(n_el):
            idx, mask_val = _ctz_and_clear_64(mask_val)
            trans1_diag[I, count] = idx

    return trans1_diag


@njit(parallel=True, fastmath=True, boundscheck=False, cache=False, nogil=True)
def get_one_body_diagonal_transitions_jit(string_list, n_el, n_orbitals):
    """
    Generate trans1_diag (a table of occupied orbitals for each string) 
    for diagonal Hamiltonian evaluation.
    
    Args:
        string_list: List of BitString128 instances representing the basis.
        n_el: Number of electrons (occupied orbitals).
        n_orbitals: Total number of orbitals.
        
    Returns:
        trans1_diag (ndarray): Int32 array of shape (n_strings, n_el).
                               trans1_diag[I, k] = k-th occupied orbital in string I.
    """
    n_strings = len(string_list)
    
    trans1_diag = np.zeros((n_strings, n_el), dtype=np.int32)

    for I in prange(n_strings):
        src_mask = string_list[I]
        
        count = 0
        for p in range(n_orbitals):
            if src_mask.get_bit(p):
                trans1_diag[I, count] = p
                count += 1
                
                if count == n_el:
                    break
                    
    return trans1_diag



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



@njit(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def get_one_body_transitions_64_jit(string_array, n_orbitals):
    """
    非対角遷移テーブルを1次元フラット配列へダイレクトに並列書き込みする
    ※ p == q（対角項）は最初から排除してカウント・格納
    """
    n_strings = string_array.shape[0]

    # --- ステップ1: 各配置の「純粋な非対角遷移（p != q）」の数を並列カウント ---
    trans1_counts = np.zeros(n_strings, dtype=np.int32)
    for I in prange(n_strings):
        mask = string_array[I]
        c = 0
        for p in range(n_orbitals):
            if (mask >> p) & 1:  # 軌道 p が占有されている
                for q in range(n_orbitals):
                    # 💡 p != q かつ 軌道 q が空いている場合のみカウント
                    if p != q and not ((mask >> q) & 1):
                        c += 1
        trans1_counts[I] = c

    # --- ステップ2: オフセット配列を構築（累積和） ---
    trans1_offsets = np.zeros(n_strings + 1, dtype=np.int64)
    for I in range(n_strings):
        trans1_offsets[I + 1] = trans1_offsets[I] + trans1_counts[I]
    
    total_trans = trans1_offsets[-1]
    
    # --- ステップ3: 最初から最終形状の1次元フラット配列を確保 ---
    trans1_flat_J = np.zeros(total_trans, dtype=np.int32)
    trans1_flat_p = np.zeros(total_trans, dtype=np.int8)
    trans1_flat_q = np.zeros(total_trans, dtype=np.int8)

    # --- ステップ4: 並列で1次元配列にダイレクト書き込み ---
    for I in prange(n_strings):
        mask = string_array[I]
        write_idx = trans1_offsets[I]

        for p in range(n_orbitals):
            if (mask >> p) & 1:
                for q in range(n_orbitals):
                    if p != q and not ((mask >> q) & 1):
                        dest_mask = mask ^ (1 << p) ^ (1 << q)
                        
                        J = binary_search_64(string_array, dest_mask)
                        
                        if J != -1:
                            trans1_flat_J[write_idx] = J
                            trans1_flat_p[write_idx] = p
                            trans1_flat_q[write_idx] = q
                            write_idx += 1
                            
    return trans1_offsets, trans1_flat_J, trans1_flat_p, trans1_flat_q



@njit(parallel=True, fastmath=True, boundscheck=False, cache=False, nogil=True)
def get_one_body_transitions_jit(string_list, string_to_idx, n_el, n_orbitals):
    """
    Generate a transition table for all one-body operators a_p^dagger a_q.
    
    Args:
        string_list: List of BitString128 instances representing the basis.
        string_to_idx: Numba Dict mapping (uint64, uint64) to int32 index.
        n_el: Number of particles 
        n_orbitals: Total number of orbitals
    """
    n_strings = len(string_list)        

    trans1_counts = np.zeros(n_strings, dtype=np.int32)
    
    estd_max_counts = n_el * (n_orbitals - n_el + 1)
    trans1_J = np.zeros((n_strings, estd_max_counts), dtype=np.int32)
    trans1_p = np.zeros((n_strings, estd_max_counts), dtype=np.int32)
    trans1_q = np.zeros((n_strings, estd_max_counts), dtype=np.int32)
    trans1_phase = np.zeros((n_strings, estd_max_counts), dtype=np.int8)

    for I in prange(n_strings):
        src_mask = string_list[I]
        
        for p in range(n_orbitals):
            # Check if orbital p is occupied in source string <I|
            if src_mask.get_bit(p):
                
                for q in range(n_orbitals):
                    # For p == q (diagonal), or if q is empty in the state after removing p
                    # Actually, <I| a_p^dagger a_q is non-zero iff:
                    # 1. p == q
                    # 2. p != q AND orbital q is empty in |I>
                    
                    if p == q:
                        # Diagonal term: <I|a_p^dagger a_p|I>
                        # Destination is the same as source, phase is always 1
                        #results.append((I, I, p, q, 1))
                        trans1_J[I, trans1_counts[I]] = I
                        trans1_p[I, trans1_counts[I]] = p
                        trans1_q[I, trans1_counts[I]] = p
                        trans1_phase[I, trans1_counts[I]] = 1
                        trans1_counts[I] += 1
                        
                    elif not src_mask.get_bit(q):
                        # Off-diagonal term: <I|a_p^dagger a_q|J>
                        dest_mask = src_mask.apply_transition(q, p)
                        dest_key = dest_mask.to_tuple()
                        
                        if dest_key in string_to_idx:
                            J = string_to_idx[dest_key]
                            # Calculate phase (-1)^n_mid
                            n_mid = src_mask.count_between(q, p)
                            phase = 1 if n_mid % 2 == 0 else -1
                            #results.append((J, I, p, q, phase))
                            trans1_J[I, trans1_counts[I]] = J
                            trans1_p[I, trans1_counts[I]] = p
                            trans1_q[I, trans1_counts[I]] = q
                            trans1_phase[I, trans1_counts[I]] = phase
                            trans1_counts[I] += 1

    max_counts = np.max(trans1_counts)
    trans1_J = trans1_J[:, :max_counts].copy()
    trans1_p = trans1_p[:, :max_counts].copy()
    trans1_q = trans1_q[:, :max_counts].copy()
    trans1_phase = trans1_phase[:, :max_counts].copy()

    return trans1_counts, trans1_J, trans1_p, trans1_q, trans1_phase



@njit#(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def get_two_body_diagonal_transitions_jit(string_list, n_el, n_orbitals):
    """
    Generate trans2_diag (occupied orbital pairs p < q) for diagonal Hamiltonian evaluation.

    Args:
        string_list: List of BitString128 instances representing the basis.
        n_el: Number of particles (electrons).
        n_orbitals: Total number of orbitals.

    Returns:
        trans2_p (ndarray): Int32 array of shape (n_strings, n_pairs).
        trans2_q (ndarray): Int32 array of shape (n_strings, n_pairs).
                            where n_pairs = n_el * (n_el - 1) // 2.
    """
    n_strings = len(string_list)
    
    # 各ストリングが持つ占有軌道のペア数は完全に固定 (n_el C 2)
    n_pairs = n_el * (n_el - 1) // 2

    # 対角項なので r=p, s=q で確定。保持する必要があるのは独立な p と q のみ
    trans2_p = np.zeros((n_strings, n_pairs), dtype=np.int32)
    trans2_q = np.zeros((n_strings, n_pairs), dtype=np.int32)

    for I in prange(n_strings):
        src_mask = string_list[I]
        
        pair_count = 0
        # 占有されている軌道のペア (p < q) を直線的に探索
        for p in range(n_orbitals):
            if src_mask.get_bit(p):
                for q in range(p + 1, n_orbitals):
                    if src_mask.get_bit(q):
                        
                        # 対角条件 (p==r かつ q==s) に相当するペアを記録
                        trans2_p[I, pair_count] = p
                        trans2_q[I, pair_count] = q
                        pair_count += 1
                        
                        # すべてのペアを回収しきったら、早期にループを抜ける
                        if pair_count == n_pairs:
                            break
            if pair_count == n_pairs:
                break

    return trans2_p, trans2_q



@njit#(parallel=True, fastmath=True, boundscheck=False, cache=True, nogil=True)
def get_two_body_transitions_jit(string_list, string_to_idx, n_el, n_orbitals):
    """
    Generate a transition table for all two-body operators a_p^dagger a_q^dagger a_s a_r.
    
    Args:
        string_list: List of BitString128 instances representing the basis.
        string_to_idx: Numba Dict mapping (uint64, uint64) to int32 index.
        n_el: Number of particles
        n_orbitals: Total number of orbitals
    """
    n_strings = len(string_list)

    trans2_counts = np.zeros(n_strings, dtype=np.int32)

    # 見積もられる最大遷移数 (p<q で抽出する組み合わせ数 × r<s で生成する組み合わせ数)
    max_pq = (n_el * (n_el - 1)) // 2
    n_vir = n_orbitals - n_el
    max_rs = ((n_vir + 2) * (n_vir + 1)) // 2
    estd_max_counts = max_pq * max_rs

    trans2_J = np.zeros((n_strings, estd_max_counts), dtype=np.int32)
    trans2_p = np.zeros((n_strings, estd_max_counts), dtype=np.int32)
    trans2_q = np.zeros((n_strings, estd_max_counts), dtype=np.int32)
    trans2_r = np.zeros((n_strings, estd_max_counts), dtype=np.int32)
    trans2_s = np.zeros((n_strings, estd_max_counts), dtype=np.int32)
    trans2_phase = np.zeros((n_strings, estd_max_counts), dtype=np.int8)

    for I in prange(n_strings):
        src_mask = string_list[I]

        # 1. p, q (引き抜く軌道) の探索
        for p in range(n_orbitals):
            if src_mask.get_bit(p):
                for q in range(p + 1, n_orbitals):
                    if src_mask.get_bit(q):

                        # 2. r, s (加える軌道) の探索
                        for r in range(n_orbitals):
                            for s in range(r + 1, n_orbitals):
                                
                                # r と s は p, q が抜けた後の状態で空でなければならない
                                # (元から空であるか、p または q と一致している必要がある)
                                if src_mask.get_bit(r) and r != p and r != q:
                                    continue
                                if src_mask.get_bit(s) and s != p and s != q:
                                    continue

                                if p == r and q == s:
                                    # 対角項: <I| a_p^dagger a_q^dagger a_q a_p |I>
                                    # 遷移先は同じ状態で、位相は必ず 1
                                    trans2_J[I, trans2_counts[I]] = I
                                    trans2_p[I, trans2_counts[I]] = p
                                    trans2_q[I, trans2_counts[I]] = q
                                    trans2_r[I, trans2_counts[I]] = r
                                    trans2_s[I, trans2_counts[I]] = s
                                    trans2_phase[I, trans2_counts[I]] = 1
                                    trans2_counts[I] += 1

                                else:
                                    # 非対角項: <I| a_p^dagger a_q^dagger a_s a_r |J>
                                    # 先ほど追加された apply_two_electron を使用
                                    dest_mask, phase = src_mask.apply_two_electron(p, q, r, s)
                                    #dest_mask, phase = src_mask.apply_two_electron(r, s, q, p)
                                    dest_key = dest_mask.to_tuple()

                                    if dest_key in string_to_idx:
                                        J = string_to_idx[dest_key]
                                        
                                        trans2_J[I, trans2_counts[I]] = J
                                        trans2_p[I, trans2_counts[I]] = p
                                        trans2_q[I, trans2_counts[I]] = q
                                        trans2_r[I, trans2_counts[I]] = r
                                        trans2_s[I, trans2_counts[I]] = s
                                        trans2_phase[I, trans2_counts[I]] = phase
                                        trans2_counts[I] += 1

    # 最大遷移数に合わせて配列を切り詰める
    max_counts = np.max(trans2_counts)
    trans2_J = trans2_J[:, :max_counts].copy()
    trans2_p = trans2_p[:, :max_counts].copy()
    trans2_q = trans2_q[:, :max_counts].copy()
    trans2_r = trans2_r[:, :max_counts].copy()
    trans2_s = trans2_s[:, :max_counts].copy()
    trans2_phase = trans2_phase[:, :max_counts].copy()

    return trans2_counts, trans2_J, trans2_p, trans2_q, trans2_r, trans2_s, trans2_phase




@njit
def generate_strings_1groups_jit(limits, n1):
    """
    Pure JIT implementation to generate RAS strings.
    No explicit type specification to avoid TypingErrors.
    """
    # 1. Generate local bit patterns
    list1 = get_combinations_64(limits[0], n1)

    # 2. Initialize an empty typed List without explicit type.
    # Numba will infer the type from the first 'append' call.
    all_strings = typed.List()
    for m1 in list1:
        # Create a new instance
        s = BitString128(uint64(0), uint64(0))

        # Apply bitmask logic
        s.low |= m1

        # The first append here locks the type of
        # all_strings to 'BitString128'
        all_strings.append(s)

    return all_strings



@njit
def generate_strings_2groups_jit(limits, n1, n2):
    """
    Pure JIT implementation to generate RAS strings.
    No explicit type specification to avoid TypingErrors.
    """
    # 1. Generate local bit patterns
    list1 = get_combinations_64(limits[0], n1)
    list2 = get_combinations_64(limits[1], n2)

    offset2 = limits[0]

    # 2. Initialize an empty typed List without explicit type.
    # Numba will infer the type from the first 'append' call.
    all_strings = typed.List()
    for m1 in list1:
        for m2 in list2:
            # Create a new instance
            s = BitString128(uint64(0), uint64(0))

            # Apply bitmask logic
            s.low |= m1
            s = _apply_shifted_mask(s, m2, offset2)

            # The first append here locks the type of
            # all_strings to 'BitString128'
            all_strings.append(s)

    return all_strings


@njit
def generate_strings_3groups_jit(limits, n1, n2, n3):
    """
    Pure JIT implementation to generate RAS strings.
    No explicit type specification to avoid TypingErrors.
    """
    # 1. Generate local bit patterns
    list1 = get_combinations_64(limits[0], n1)
    list2 = get_combinations_64(limits[1], n2)
    list3 = get_combinations_64(limits[2], n3)

    offset2 = limits[0]
    offset3 = limits[0] + limits[1]

    # 2. Initialize an empty typed List without explicit type.
    # Numba will infer the type from the first 'append' call.
    all_strings = typed.List()
    for m1 in list1:
        for m2 in list2:
            for m3 in list3:
                # Create a new instance
                s = BitString128(uint64(0), uint64(0))

                # Apply bitmask logic
                s.low |= m1
                s = _apply_shifted_mask(s, m2, offset2)
                s = _apply_shifted_mask(s, m3, offset3)

                # The first append here locks the type of
                # all_strings to 'BitString128'
                all_strings.append(s)

    return all_strings



@njit
def generate_strings_4groups_jit(limits, n1, n2, n3, n4):
    """
    Pure JIT implementation to generate RAS strings.
    No explicit type specification to avoid TypingErrors.
    """
    # 1. Generate local bit patterns
    list1 = get_combinations_64(limits[0], n1)
    list2 = get_combinations_64(limits[1], n2)
    list3 = get_combinations_64(limits[2], n3)
    list4 = get_combinations_64(limits[3], n4)

    offset2 = limits[0]
    offset3 = limits[0] + limits[1]
    offset4 = limits[0] + limits[1] + limits[2]

    # 2. Initialize an empty typed List without explicit type.
    # Numba will infer the type from the first 'append' call.
    all_strings = typed.List()
    for m1 in list1:
        for m2 in list2:
            for m3 in list3:
                for m4 in list4:
                    # Create a new instance
                    s = BitString128(uint64(0), uint64(0))
                    
                    # Apply bitmask logic
                    s.low |= m1
                    s = _apply_shifted_mask(s, m2, offset2)
                    s = _apply_shifted_mask(s, m3, offset3)
                    s = _apply_shifted_mask(s, m4, offset4)
                    
                    # The first append here locks the type of
                    # all_strings to 'BitString128'
                    all_strings.append(s)

    return all_strings


@njit#(parallel=True)
def get_trans_boundaries_for_ormas2_jit(
        len_a_strings,
        det_allowed,
        det_ll,
        det_num,
        trans_counts,
        trans_J
):
    '''
    # Offset and number of determinants
    '''
    len_b_dist = det_allowed.shape[1]
    max_counts = np.max(trans_counts)

    II_ll  = np.zeros((len_a_strings, max_counts, len_b_dist), dtype=np.int64)
    II_num = np.zeros((len_a_strings, max_counts, len_b_dist), dtype=np.int64)
    JI_ll  = np.zeros((len_a_strings, max_counts, len_b_dist), dtype=np.int64)
    JI_num = np.zeros((len_a_strings, max_counts, len_b_dist), dtype=np.int64)

    for I_a in prange(len_a_strings):

        for i_trans in range(trans_counts[I_a]):
            J_a = trans_J[I_a, i_trans]

            for _i in range(len_b_dist):
                both_allowed = det_allowed[I_a, _i] * det_allowed[J_a, _i]
                if both_allowed:
                    II_ll [I_a, i_trans, _i] = det_ll [I_a, _i]
                    II_num[I_a, i_trans, _i] = det_num[I_a, _i]
                    JI_ll [I_a, i_trans, _i] = det_ll [J_a, _i]
                    JI_num[I_a, i_trans, _i] = det_num[J_a, _i]

    return II_ll, II_num, JI_ll, JI_num
