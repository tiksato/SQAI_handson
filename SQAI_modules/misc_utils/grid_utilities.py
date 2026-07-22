import numpy as np
from scipy.special import erf


def get_Coulomb(R, Z):
    return -Z / np.sqrt(R * R)


def get_Coulomb_gradient(R, Z):
    return +Z / (R * R)


def get_soft_Coulomb(R, Z, d):
    return -Z / np.sqrt(R * R + d)


def get_soft_Coulomb_gradient(R, Z, d):
    return Z * R / np.sqrt(R * R + d) ** 3


def get_erf_term(ZR, mu, small=1e-4, data_type=np.float64):
    ZRa = np.asarray([ZR], dtype=data_type).flatten()
    muZR = mu * ZRa
    # erf_term = np.where(np.abs(ZRa) < small,
    #                    2*mu/np.sqrt(np.pi)*(1-muZR**2/3+muZR**4/10-muZR**6/42+muZR**8/216),
    #                    erf(muZR)/ZRa)
    erf_term = np.zeros_like(ZRa)
    for i in range(len(ZRa)):
        if np.abs(ZRa[i]) < small:
            erf_term[i] = (
                2
                * mu
                / np.sqrt(np.pi)
                * (
                    1
                    - muZR[i] ** 2 / 3
                    + muZR[i] ** 4 / 10
                    - muZR[i] ** 6 / 42
                    + muZR[i] ** 8 / 216
                )
            )
        else:
            erf_term[i] = erf(muZR[i]) / ZRa[i]
    return erf_term


def get_erf_term_small(ZR, mu):
    muZR = mu * ZR
    print("small:", ZR)
    return (
        2
        * mu
        / np.sqrt(np.pi)
        * (1 - muZR**2 / 3 + muZR**4 / 10 - muZR**6 / 42 + muZR**8 / 216)
    )


def get_erf_term_large(ZR, mu):
    print("large:", ZR)
    return erf(mu * ZR) / ZR


def get_erfgau(R, Z=1.0, mu=1.25, data_type=np.float64):
    Ra = np.asarray([R], dtype=data_type).flatten()
    absR = np.abs(Ra)
    c_erfgau = 0.9230 + 1.568 * mu
    a_erfgau = 0.2411 + 1.405 * mu

    ZR = Z * absR
    erf_term = get_erf_term(ZR, mu, small=1e-4, data_type=data_type)
    return -(Z**2) * (c_erfgau * np.exp(-((a_erfgau * ZR) ** 2)) + erf_term)
