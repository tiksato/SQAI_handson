import numpy as np
from numba import njit, uint64, types
from numba.experimental import jitclass

# Specification for JIT class fields
spec = [
    ('low', uint64),
    ('high', uint64),
]

@jitclass(spec)
class BitString128:
    def __init__(self, low=0, high=0):
        """
        Initialize a 128-bit bitmask using two 64-bit unsigned integers.
        Supports up to 128 orbitals (e.g., for 125-orbital RASCI).
        """
        self.low = uint64(low)
        self.high = uint64(high)

    def set_bit(self, n):
        """Set the n-th bit (0-127) to 1."""
        if n < 64:
            self.low |= (uint64(1) << uint64(n))
        else:
            self.high |= (uint64(1) << uint64(n - 64))

    def get_bit(self, n):
        """Return the state of the n-th bit (0 or 1)."""
        if n < 64:
            return (self.low >> uint64(n)) & uint64(1)
        else:
            return (self.high >> uint64(n - 64)) & uint64(1)

    def apply_transition(self, p, q):
        """
        Perform a_p^dagger a_q operation (annihilate q, then create p).
        Returns a new BitString128 instance.
        """
        new_low = self.low
        new_high = self.high
        
        # Annihilation of orbital q
        if q < 64: 
            new_low ^= (uint64(1) << uint64(q))
        else:      
            new_high ^= (uint64(1) << uint64(q - 64))
            
        # Creation of orbital p
        if p < 64: 
            new_low |= (uint64(1) << uint64(p))
        else:      
            new_high |= (uint64(1) << uint64(p - 64))
            
        return BitString128(new_low, new_high)

    def count_between(self, p, q):
        """
        Count the number of set bits (electrons) between orbital p and q.
        Essential for fermion phase calculation: (-1)^count.
        """
        idx_min = min(p, q)
        idx_max = max(p, q)
        
        # If orbitals are adjacent or identical, no orbitals in between
        if idx_max - idx_min <= 1:
            return 0
        
        b_min = idx_min // 64
        b_max = idx_max // 64
        
        count = 0
        if b_max == 0: 
            # Case 1: Both orbitals are in the Low block
            mask = self._make_mask(idx_min + 1, idx_max - 1)
            count = self._popcount(self.low & mask)
        elif b_min == 1: 
            # Case 2: Both orbitals are in the High block
            mask = self._make_mask(idx_min - 64 + 1, idx_max - 64 - 1)
            count = self._popcount(self.high & mask)
        else: 
            # Case 3: Transition spans across Low and High blocks
            # Mask from (idx_min + 1) to 63 in Low block
            mask_l = ~((uint64(1) << uint64(idx_min + 1)) - 1)
            # Mask from 0 to (idx_max - 1) in High block
            mask_h = (uint64(1) << uint64(idx_max - 64)) - 1
            count = self._popcount(self.low & mask_l) + self._popcount(self.high & mask_h)
            
        return count

    def _make_mask(self, l, h):
        """Generate a bitmask where bits from index l to h are 1 (inclusive)."""
        return ((uint64(1) << uint64(h + 1)) - 1) & ~((uint64(1) << uint64(l)) - 1)

    def _popcount(self, n):
        n = uint64(n)
        c = 0
        while n > 0:
            n &= (n - uint64(1))  # 引く数字の「1」もuint64にする
            c += 1
        return c

    def to_tuple(self):
        """Convert to tuple for use as a dictionary key (BitString128 itself is mutable)."""
        return (uint64(self.low), uint64(self.high))

    def is_equal(self, other):
        """Check equality between two BitString128 instances."""
        return self.low == other.low and self.high == other.high

    def to_bitstring(self, length):
        """Returns a string representation of the bits (e.g., '00101...')"""
        res = ""
        for i in range(length):
            res += "1" if self.get_bit(i) else "0"
        return res

    def count_strictly_below(self, idx):
        """
        Count the number of set bits (electrons) strictly below orbital idx.
        Essential for step-by-step fermionic phase evaluation.
        """
        if idx <= 0:
            return 0
            
        if idx < 64:
            mask = (uint64(1) << uint64(idx)) - uint64(1)
            return self._popcount(self.low & mask)
        else:
            mask = (uint64(1) << uint64(idx - 64)) - uint64(1)
            return self._popcount(self.low) + self._popcount(self.high & mask)

    def apply_two_electron(self, p, q, r, s):
        """
        Perform a_p^dagger a_q^dagger a_s a_r operation on the bra <I|.
        Order of action: p(annihilate) -> q(annihilate) -> s(create) -> r(create)
        """
        temp = BitString128(self.low, self.high)
        parity_count = 0

        # 1. Annihilate p (a_p^dagger)
        parity_count += temp.count_strictly_below(p)
        if p < 64:
            temp.low ^= (uint64(1) << uint64(p))  # |= ではなく ^= (XOR) で 1 -> 0 に！
        else:
            temp.high ^= (uint64(1) << uint64(p - 64))

        # 2. Annihilate q (a_q^dagger)
        parity_count += temp.count_strictly_below(q)
        if q < 64:
            temp.low ^= (uint64(1) << uint64(q))
        else:
            temp.high ^= (uint64(1) << uint64(q - 64))

        # 3. Create s (a_s)
        parity_count += temp.count_strictly_below(s)
        if s < 64:
            temp.low ^= (uint64(1) << uint64(s))  # 0 -> 1
        else:
            temp.high ^= (uint64(1) << uint64(s - 64))

        # 4. Create r (a_r)
        parity_count += temp.count_strictly_below(r)
        if r < 64:
            temp.low ^= (uint64(1) << uint64(r))
        else:
            temp.high ^= (uint64(1) << uint64(r - 64))

        phase = 1 if (parity_count % 2 == 0) else -1
        return temp, phase

#    def apply_two_electron(self, p, q, r, s):
#        """
#        Perform a_p^dagger a_q^dagger a_s a_r operation step-by-step.
#        Returns a tuple of (new_BitString128, phase).
#        Assumes the caller has already verified the transition is physically valid 
#        (e.g., r and s are occupied).
#        """
#        # Create a temporary mutable state to track intermediate bit changes
#        temp = BitString128(self.low, self.high)
#        parity_count = 0
#        
#        # 1. Annihilate r (a_r)
#        parity_count += temp.count_strictly_below(r)
#        if r < 64:
#            temp.low ^= (uint64(1) << uint64(r))
#        else:
#            temp.high ^= (uint64(1) << uint64(r - 64))
#            
#        # 2. Annihilate s (a_s)
#        parity_count += temp.count_strictly_below(s)
#        if s < 64:
#            temp.low ^= (uint64(1) << uint64(s))
#        else:
#            temp.high ^= (uint64(1) << uint64(s - 64))
#            
#        # 3. Create q (a_q^dagger)
#        parity_count += temp.count_strictly_below(q)
#        if q < 64:
#            temp.low |= (uint64(1) << uint64(q))
#        else:
#            temp.high |= (uint64(1) << uint64(q - 64))
#            
#        # 4. Create p (a_p^dagger)
#        parity_count += temp.count_strictly_below(p)
#        if p < 64:
#            temp.low |= (uint64(1) << uint64(p))
#        else:
#            temp.high |= (uint64(1) << uint64(p - 64))
#            
#        phase = 1 if (parity_count % 2 == 0) else -1
#        return temp, phase
    
@njit
def test_bitstring():
    print("--- Test Start ---")
    
    # 1. Initialization and bit setting
    s = BitString128(0, 0)
    s.set_bit(10)  # Low block (orbital 10)
    s.set_bit(100) # High block (orbital 100)
    
    print("Bit 10:", s.get_bit(10))   # Expected: 1
    print("Bit 100:", s.get_bit(100)) # Expected: 1
    print("Bit 50:", s.get_bit(50))   # Expected: 0
    
    # 2. Transition test: a_120^\dagger a_10
    # Move electron from orbital 10 to 120
    s_new = s.apply_transition(120, 10)
    print("After Transition (10->120):")
    print("Bit 10:", s_new.get_bit(10))   # Expected: 0
    print("Bit 120:", s_new.get_bit(120)) # Expected: 1
    
    # 3. Phase test: Transition between 0 and 120
    # Setup: One electron exists at orbital 100
    base = BitString128(0, 0)
    base.set_bit(100) 
    
    # Orbitals between 0 and 120 include orbital 100 -> count = 1
    n_mid = base.count_between(0, 120)
    phase = 1.0 if n_mid % 2 == 0 else -1.0
    print("Count between 0 and 120:", n_mid) # Expected: 1
    print("Phase:", phase)                   # Expected: -1.0
    
    # 4. Adjacent orbitals test
    print("Count between 10 and 11:", base.count_between(10, 11)) # Expected: 0
    
    # 5. Tuple conversion for dictionary mapping
    t = s_new.to_tuple()
    print("Tuple representation:", t)
    
    print("--- Test End ---")

#test_bitstring()
