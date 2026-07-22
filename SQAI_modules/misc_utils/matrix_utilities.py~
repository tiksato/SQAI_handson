import numpy as np
from scipy.linalg import eig
from pyscf import lib as lib_pyscf

def canonical_orthonormalization(overlap):
    S = overlap
    eS, US = np.linalg.eigh(S)
    X = np.einsum("ij,j->ij", US, 1 / np.sqrt(eS))
    Xinv = np.linalg.inv(X)
    return (X, Xinv)


#def arnoldi(fun, x0, arnoldi_order):
#    n = x0.shape[0]
#    m = arnoldi_order
#    if m > n:
#        raise ValueError
#    h = np.zeros((m, m), dtype=np.complex128)
#    v = np.zeros((n, m), dtype=np.complex128)
#
#    v0 = x0 / np.linalg.norm(x0)
#    v[:, 0] = v0
#
#    k = 0
#    for k in range(m):
#        k = k
#        w = fun(v[:, k])
#        for j in range(k + 1):
#            v_j = v[:, j]
#            h[j, k] = np.conj(v_j.T) @ w
#            w = w - h[j, k] * v_j
#
#        if k + 1 < m:
#            h[k + 1, k] = np.linalg.norm(w)
#            v[:, k + 1] = w / h[k + 1, k]
#
#    ritz_val, ritz_vec = np.linalg.eig(h)
#    eig_vec = v @ ritz_vec
#    return (ritz_val, eig_vec)


def arnoldi(get_sigma, n_dim, n_ev,
            v_init=None, m_krylov=20, tol=1e-8):
    """
    Arnoldi iteration for finding the lowest real-part eigenvalues.
    
    Parameters:
    -----------
    get_sigma : function
        User-defined routine that computes the sigma-vector: sigma = H * c.
        This is the core of the Direct CI method.
    n_dim : int
        Dimension of the CI subspace (number of determinants/CSFs).
    n_ev : int
        Number of desired eigenvalues (eigenvectors).
    v_init: np.ndarray(np.complex128)
        Initial vector
    m_krylov : int
        Maximum dimension of the Krylov subspace.
    tol : float
        Convergence tolerance for the residual.
    """
    
    # 1. Initialize the Krylov subspace basis
    # Using a random starting vector; ensure reproducibility with a seed if needed.
    if v_init is None:
        v = (1+0j)*np.random.rand(n_dim)
    else:
        v = (1+0j)*v_init.copy()
    v /= np.linalg.norm(v)
    
    V = np.zeros((n_dim, m_krylov + 1), dtype=np.complex128)
    H_hessenberg = np.zeros((m_krylov, m_krylov), dtype=np.complex128)
    V[:, 0] = v

    actual_m = m_krylov

    for j in range(m_krylov):
        # Apply the Hamiltonian operator (Direct CI sigma-vector generation)
        w = get_sigma(V[:, j])
        
        # Modified Gram-Schmidt (MGS) orthonormalization
        for i in range(j + 1):
            H_hessenberg[i, j] = np.vdot(V[:, i], w)
            w -= H_hessenberg[i, j] * V[:, i]
        
        h_next = np.linalg.norm(w)
        
        # Check for early convergence (unlikely in large CI spaces)
        if h_next < tol:
            actual_m = j + 1
            break
        
        # Store the next basis vector
        if j + 1 < m_krylov:
            H_hessenberg[j + 1, j] = h_next
            V[:, j + 1] = w / h_next

    # 2. Solve the eigenvalue problem for the small Hessenberg matrix
    # Note: For Hermitian H, this becomes a symmetric tridiagonal matrix.
    reduced_h = H_hessenberg[:actual_m, :actual_m]
    eigvals, eigvecs_small = eig(reduced_h)
    
    # 3. Sort by the real part of eigenvalues (ascending order)
    # This selects the lowest energy states.
    idx = np.argsort(np.real(eigvals))
    selected_indices = idx[:n_ev]
    
    final_eigvals = eigvals[selected_indices]
    
    # 4. Transform Ritz vectors back to the full CI space
    # Ritz vector = V_m * y_i
    final_eigvecs = V[:, :actual_m] @ eigvecs_small[:, selected_indices]
    
    return final_eigvals, final_eigvecs


def davidson(tot_dim, get_sigma, precond,
             x_init = None,
             nroots = 1,
             tol = 1E-12,
             verbose=0):
    
    if x_init is None:
        x_init = np.zeros(tot_dim, dtype=np.float64)
        x_init[0] = 1.0

    eigenvalues, eigenvectors = lib_pyscf.davidson(
        get_sigma,
        [x_init],
        precond,
        tol=tol,
        max_cycle=100,
        max_space=20,
        nroots=nroots,
        verbose=verbose)

    return eigenvalues, eigenvectors



def davidson_nosym(tot_dim, get_sigma, precond,
                   pick = None,
                   x_init = None,
                   nroots = 1,
                   tol = 1E-12,
                   verbose=0):

    if pick is None:
        
        def pick_by_real_part(w, v, nroots, envs):
            idx = np.argsort(w.real)[:nroots]
            return w[idx], v[:, idx], idx
        
        pick = pick_by_real_part
    
    if x_init is None:
        x_init = np.zeros(tot_dim, dtype=np.float64)
        x_init[0] = 1.0

    eigenvalues, eigenvectors = lib_pyscf.davidson_nosym(
        get_sigma,
        [x_init],
        precond,
        nroots = nroots,
        pick = pick, 
        tol = tol,
        max_cycle = 100,
        max_space = 20,
        verbose = verbose)

    return eigenvalues, eigenvectors


def get_operator_matrix(tot_dim, func_mat_vec):

    ham = np.zeros((tot_dim, tot_dim),
                   dtype=np.complex128)

    eye_vec = np.zeros(tot_dim,
                       dtype=np.complex128)

    for i in range(tot_dim):
        eye_vec[i] = 1.0
        ham[i,:] = func_mat_vec(eye_vec)
        eye_vec[i] = 0.0

    return np.ascontiguousarray(ham.T)



def get_operator_diagonal(tot_dim, func_mat_vec):

    hdiag = np.zeros(tot_dim,
                     dtype=np.complex128)

    eye_vec = np.zeros(tot_dim,
                       dtype=np.complex128)

    for i in range(tot_dim):
        eye_vec[i] = 1.0
        sigma_i = func_mat_vec(eye_vec)
        hdiag[i] = sigma_i[i]
        eye_vec[i] = 0.0

    return hdiag

#def arnoldi_optimization(tot_dim, get_sigma):
#    x_init = np.zeros(tot_dim, dtype=np.complex128)
#    x_init[0] = 1.0
#    
#    eigenvalues, eigenvectors = arnoldi_rasci(
#        get_sigma, 
#        tot_dim, 
#        1, 
#        x_init, 
#        m_krylov=40)
#    
#    return eigenvalues, eigenvectors

