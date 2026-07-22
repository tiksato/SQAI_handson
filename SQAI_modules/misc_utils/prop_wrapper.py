import numpy as np


class prop_wrapper:
    def __init__(self, imaginary, dtime, TDHF, Laser=None):
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
        self.TDHF = TDHF
        self.n_fun = TDHF.n_fun
        self.n_bas = TDHF.n_bas
        self.n_grid = TDHF.n_grid
        self.mval_list = TDHF.mval_list
        self.L_size = self.n_fun * self.n_bas

        mval = self.mval_list[0]
        self.L0mat = self.t_fac * TDHF.basis[mval].H0mat

        self.V2 = np.zeros((self.n_fun, self.n_fun, self.n_grid), dtype=np.complex128)
        self.fmat = np.zeros((self.n_fun, self.n_fun), dtype=np.complex128)

    def prodH0(self, u):
        wfn = u.reshape(self.n_fun, self.n_bas)
        hwfn = self.TDHF.prodH0(wfn)
        return hwfn.flatten()

    def prodH(self, u, time):
        wfn = u.reshape(self.n_fun, self.n_bas)
        hwfn = self.TDHF.prodH0(wfn)
        if not self.imaginary:
            fval = self.Laser.vector_potential(time)
            hwfn += fval * (self.TDHF.prodV1(wfn))
        return hwfn.flatten()

    def prodF(self, u, uin, time, initial=True, qproj=True):
        wfnin = uin.reshape(self.n_fun, self.n_bas)
        wfn = u.reshape(self.n_fun, self.n_bas)
        hwfn = self.TDHF.prodH0(wfn)
        if not self.imaginary:
            fval = self.Laser.vector_potential(time)
            hwfn += fval * (self.TDHF.prodV1(wfn))

        if initial:
            self.TDHF.calc_potential(wfnin)
            self.V2[:, :, :] = self.TDHF.Pot_ang.copy()
        gwfn = self.TDHF.prodG(wfn, self.V2)
        #########
        # gwfn *= 0
        #########

        fwfn = hwfn + gwfn
        if initial:
            self.fmat = self.TDHF.get_int1e(wfn, fwfn)

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
        h0wfn = self.TDHF.prodH0(wfn)

        self.TDHF.calc_potential(wfn)
        Gwfn = self.TDHF.prodG(wfn)

        if self.imaginary:
            Fmat = self.TDHF.get_int1e(wfn, h0wfn + Gwfn)
            dFmat = Fmat - self.Fmat0

            G0wfn = self.TDHF.prodG(wfn, self.V2)
            dGwfn = Gwfn - G0wfn

            wfn1 = dGwfn - dFmat.T @ wfn
        else:
            fval0 = self.Laser.vector_potential(self.time_left)
            fval = self.Laser.vector_potential(time)
            tmpV1 = self.TDHF.prodV1(wfn)
            v1wfn = fval * tmpV1
            dv1wfn = (fval - fval0) * tmpV1

            Fmat = self.TDHF.get_int1e(wfn, h0wfn + v1wfn + Gwfn)
            dFmat = Fmat - self.Fmat0

            G0wfn = self.TDHF.prodG(wfn, self.V2)
            dGwfn = Gwfn - G0wfn

            wfn1 = dv1wfn + dGwfn - dFmat.T @ wfn

        return self.t_fac * wfn1.flatten()
