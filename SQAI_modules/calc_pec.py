import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
#sys.path.insert(0, str(Path.cwd().parent / "modules"))

import time
import math
import numpy as np
from pyscf import gto

from misc_utils.pyscf_tools import get_integrals_rhf
from misc_utils.pyscf_tools import davidson_restricted_fullci_mask as davidson_pyscf
from misc_utils.matrix_utilities import davidson
from ormas_tools.rasci import Full_CI, RAS_CI, RHF_CI

def get_orbital_info(mol):

    n_fun = mol.nao
    n_core = 0
    n_act = n_fun - n_core
    n_elec = np.array(mol.nelec) - n_core
    
    return n_core, n_act, n_elec

    
def CI_Davidson(objCI, e_core, h1eff, h2_phys):
    # Direct-CI Davidson diagonalization
    print("Davidson diagonalization...")
    t0 = time.time()

    ##############################################
    def my_get_sigma(x):
        return objCI.h_prod(h1eff, h2_phys, x).real
    ##############################################

    hdiag = objCI.calc_hdiag(h1eff, h2_phys).real
    def my_precond(dx, e, x0, small=1e-3):
        denom = hdiag - e
        return dx * denom / (denom**2 + small)

    E1, U1 = davidson(objCI.total_dim,
                      my_get_sigma,
                      my_precond,
                      tol = 1E-6,
                      verbose=0)
    print(f" ... Done: {time.time()-t0:.2f} sec.")
    print(f"CI energy: {E1 + e_core}")
    return E1 + e_core, U1

def get_mol_list(R_list, basis):
    
    mol_list = []
    for R_H in R_list:
        mol = gto.M(
            atom=f'''
            H {-5*R_H/2} 0 0
            H {-3*R_H/2} 0 0
            H   {-R_H/2} 0 0
            H    {R_H/2} 0 0
            H  {3*R_H/2} 0 0
            H  {5*R_H/2} 0 0
            ''',
            basis=basis, 
            symmetry=False,
            verbose=0
        )
        mol_list.append(mol)
        
    return mol_list
        
def fci_pec(R_list, basis='sto-3g', num_threads=1):

    # Generate list of mol objects
    mol_list = get_mol_list(R_list, basis)
    n_core, n_act, n_elec = get_orbital_info(mol_list[0])
    print("n_core =", n_core)
    print("n_act  =", n_act)
    print("n_elec =", n_elec)

    # CI instance generation
    print("Constructing CI object...")
    t0 = time.time()
    myCI = Full_CI(
        n_elec = n_elec,
        n_orb = n_act, 
        num_threads = num_threads,
    )
    print(f" ... Done: {time.time()-t0:.2f} sec.")
    print(f"CI dimension: {myCI.total_dim}")
    print("String distribution over occupation groups:")
    print(myCI.mat_num_str)

    # CI diagonalization for list of mol objects
    return calc_for_mol_list(mol_list, myCI)



def rasci_pec(R_list, basis='sto-3g', num_threads=1):

    # Generate list of mol objects
    mol_list = get_mol_list(R_list, basis)
    n_core, n_act, n_elec = get_orbital_info(mol_list[0])
    print("n_core =", n_core)
    print("n_act  =", n_act)
    print("n_elec =", n_elec)

    # CI instance generation
    print("Constructing CI object...")
    t0 = time.time()
    n_RAS1 = 0
    n_RAS2 = sum(n_elec)
    n_RAS3 = n_act - n_RAS1 - n_RAS2
    myCI = RAS_CI(
        n_elec = n_elec,
        n_orb_RAS = (n_RAS1, n_RAS2, n_RAS3), 
        max_hole_RAS1 = 0, 
        max_elec_RAS3 = 2,
        num_threads = num_threads,
    )
    print(f" ... Done: {time.time()-t0:.2f} sec.")
    print(f"CI dimension: {myCI.total_dim}")
    print("String distribution over occupation groups:")
    print(myCI.mat_num_str)

    # CI diagonalization for list of mol objects
    return calc_for_mol_list(mol_list, myCI)



def hfci_pec(R_list, max_rank, basis='sto-3g', num_threads=1):

    # Generate list of mol objects
    mol_list = get_mol_list(R_list, basis)
    n_core, n_act, n_elec = get_orbital_info(mol_list[0])
    print("n_core =", n_core)
    print("n_act  =", n_act)
    print("n_elec =", n_elec)

    # CI instance generation
    print("Constructing CI object...")
    t0 = time.time()
    myCI = RHF_CI(
        n_elec = n_elec,
        n_orb = n_act, 
        max_rank = max_rank, 
        num_threads = num_threads,
    )
    print(f" ... Done: {time.time()-t0:.2f} sec.")
    print(f"CI dimension: {myCI.total_dim}")
    print("String distribution over occupation groups:")
    print(myCI.mat_num_str)

    # CI diagonalization for list of mol objects
    return calc_for_mol_list(mol_list, myCI)

    
def calc_for_mol_list(mol_list, objCI):


    max_dim_ormas     =  200000000 # 0.2 billion
    if objCI.total_dim > max_dim_ormas:
        print("##### WARING: CI dimension too large! #####")


    Ene_list = []
    CIC_list = []

    for i_mol, mol in enumerate(mol_list):
        #print(f"\n##### i_mol = {i_mol}/{len(mol_list)} #####")

        #print("Running RHF to obtain MO integrals...")
        t0 = time.time()
        n_core, n_act, n_elec = get_orbital_info(mol)
        e_core, h1eff, h2_phys = get_integrals_rhf(
            mol, 
            n_core, 
            n_act, 
            sum(n_elec),
        )
        #print(f" ... Done: {time.time()-t0:.2f} sec.")


        Ene, CIC = CI_Davidson(objCI, e_core, h1eff, h2_phys)
        print(f"{i_mol:5d} {Ene:12.8f}")
        Ene_list.append(Ene)
        CIC_list.append(CIC.copy())

    return np.asarray(Ene_list), CIC_list
        
