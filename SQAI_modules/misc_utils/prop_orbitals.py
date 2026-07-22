import numpy as np


class prop_orbitals:
    def __init__(self, imaginary, dtime, basis, Laser=None):
        self.time_left = 0.0
        self.time_mid = 0.0
        self.time_right = 0.0
        self.time_14 = 0.0
        self.time_34 = 0.0

        self.imaginary = imaginary
        if imaginary:
            self.t_fac = -1
        else:
            self.t_fac = -1j

        self.Laser = Laser
        self.dtime = dtime
        self.basis = basis
        self.n_fun = basis.n_fun
        self.n_bas = basis.n_bas
        self.n_grid = basis.n_grid
        self.L_size = self.n_fun * self.n_bas
        self.L0mat = self.t_fac * basis.H0mat

        #        self.V2 = np.zeros((self.n_fun, self.n_fun, self.n_grid),
        #                           dtype=np.complex128)
        self.fmat = np.zeros((self.n_fun, self.n_fun), dtype=np.complex128)

    def prodH0(self, u):
        wfn = u.reshape(self.n_fun, self.n_bas)
        hwfn = self.basis.prodH0(wfn)
        return hwfn.flatten()

    def prodH(self, u, time):
        wfn = u.reshape(self.n_fun, self.n_bas)
        hwfn = self.basis.prodH0(wfn)
        if not self.imaginary:
            fval = self.Laser.vector_potential(time)
            hwfn += fval * (self.basis.prodV1(wfn))
        return hwfn.flatten()

    def prodF(self, u, uin, time, initial=True, qproj=True):
        wfnin = uin.reshape(self.n_fun, self.n_bas)
        wfn = u.reshape(self.n_fun, self.n_bas)
        hwfn = self.basis.prodH0(wfn)
        if not self.imaginary:
            fval = self.Laser.vector_potential(time)
            hwfn += fval * (self.basis.prodV1(wfn))

        if initial:
            self.basis.calc_potential(wfnin)
            self.basis.calc_meanfield(self.D2iD1)

        gwfn = self.basis.prodV2(wfn)
        #########
        # gwfn *= 0
        #########

        fwfn = hwfn + gwfn
        if initial:
            self.fmat = self.basis.get_int1e(wfn, fwfn)

        #########
        if qproj:
            fwfn += -self.fmat.T @ wfn
        #########

        return fwfn.flatten()

    def Lxpy(self, u, uin, time, Ffac=1.0, initial=True):
        Fu = self.prodF(u, uin, time, initial)
        return u + (Ffac * self.dtime * self.t_fac) * Fu

    def getW(self, u, time):
        wfn = u.reshape(self.n_fun, self.n_bas)
        h0wfn = self.basis.prodH0(wfn)

        self.basis.calc_potential(wfn)
        Gwfn = self.basis.prodG(wfn)

        if self.imaginary:
            Fmat = self.basis.get_int1e(wfn, h0wfn + Gwfn)
            dFmat = Fmat - self.Fmat0

            G0wfn = self.basis.prodG(wfn, self.V2)
            dGwfn = Gwfn - G0wfn

            wfn1 = dGwfn - dFmat.T @ wfn
        else:
            fval0 = self.Laser.vector_potential(self.time_left)
            fval = self.Laser.vector_potential(time)
            tmpV1 = self.basis.prodV1(wfn)
            v1wfn = fval * tmpV1
            dv1wfn = (fval - fval0) * tmpV1

            Fmat = self.basis.get_int1e(wfn, h0wfn + v1wfn + Gwfn)
            dFmat = Fmat - self.Fmat0

            G0wfn = self.basis.prodG(wfn, self.V2)
            dGwfn = Gwfn - G0wfn

            wfn1 = dv1wfn + dGwfn - dFmat.T @ wfn

        return self.t_fac * wfn1.flatten()
