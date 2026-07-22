import numpy as np
from numba import njit, uint64

# 256ビット(uint64 x 4)の配列で、指定した軌道(0〜255)に電子がいるか確認する関数
@njit(fastmath=True)
def test_bit_256(bit_array, orb_index):
    # どのuint64ブロックか (orb_index // 64)
    block_idx = orb_index >> 6
    # ブロック内の何ビット目か (orb_index % 64)
    bit_pos = orb_index & 63
    
    # ビットマスクを作ってAND条件で判定
    return (bit_array[block_idx] & (uint64(1) << uint64(bit_pos))) != 0

# 256ビット(uint64 x 4)の配列に、電子を配置(1を立てる)する関数
@njit(fastmath=True)
def set_bit_256(bit_array, orb_index):
    block_idx = orb_index >> 6
    bit_pos = orb_index & 63
    bit_array[block_idx] |= (uint64(1) << uint64(bit_pos))

# 電子を消す(0にする)関数
@njit(fastmath=True)
def clear_bit_256(bit_array, orb_index):
    block_idx = orb_index >> 6
    bit_pos = orb_index & 63
    bit_array[block_idx] &= ~(uint64(1) << uint64(bit_pos))
