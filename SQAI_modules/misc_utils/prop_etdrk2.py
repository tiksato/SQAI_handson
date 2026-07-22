import numpy as np
from tqdm.notebook import tqdm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, splu, gmres

from .prop_wrapper import prop_wrapper


class prop_etdrk2(prop_wrapper):
    def __init__(self, imaginary, dtime, TDHF, Laser=None):
        super().__init__(imaginary=imaginary, dtime=dtime, TDHF=TDHF, Laser=Laser)
        self.setup()

    def setup(self):
        self.Bop1 = LinearOperator(
            shape=(self.L_size, self.L_size),
            matvec=lambda u: self.Lxpy(u, u, self.time_left, Ffac=-0.25, initial=False),
        )
        self.Bop2 = LinearOperator(
            shape=(self.L_size, self.L_size),
            matvec=lambda u: self.Lxpy(u, u, self.time_left, Ffac=-0.50, initial=False),
        )
        B0mat1 = csc_matrix(
            np.eye(self.n_bas) - (0.25 * self.dtime) * self.L0mat, dtype=np.complex128
        )
        B0mat2 = csc_matrix(
            np.eye(self.n_bas) - (0.50 * self.dtime) * self.L0mat, dtype=np.complex128
        )
        self.LUB0mat1 = splu(B0mat1)
        self.LUB0mat2 = splu(B0mat2)

        self.PC1 = LinearOperator(
            (self.L_size, self.L_size),
            lambda u: self.LUB0mat1.solve(u.reshape(self.n_fun, self.n_bas).T).T,
        )
        self.PC2 = LinearOperator(
            (self.L_size, self.L_size),
            lambda u: self.LUB0mat2.solve(u.reshape(self.n_fun, self.n_bas).T).T,
        )

    def getW(self, u, time):
        wfn = u.reshape(self.n_fun, self.n_bas)
        h0wfn = self.TDHF.prodH0(wfn)

        self.TDHF.calc_potential(wfn)
        gwfn = self.TDHF.prodG(wfn)

        g0wfn = self.TDHF.prodG(wfn, self.V2)
        dgwfn = gwfn - g0wfn

        if self.imaginary:
            fmat = self.TDHF.get_int1e(wfn, h0wfn + gwfn)
            dfmat = fmat - self.fmat
            W1 = dgwfn - dfmat.T @ wfn
        else:
            fval0 = self.Laser.vector_potential(self.time_left)
            fval = self.Laser.vector_potential(time)
            dwfn = self.TDHF.prodV1(wfn)
            v1wfn = fval * dwfn
            dv1wfn = (fval - fval0) * dwfn

            fmat = self.TDHF.get_int1e(wfn, h0wfn + v1wfn + gwfn)
            dfmat = fmat - self.fmat
            W1 = dv1wfn + dgwfn - dfmat.T @ wfn

        return self.t_fac * W1.flatten()

    def prop1_etdrk2(self, u, time, dtime=None):
        self.time_left = time
        self.time_mid = time + 0.5 * self.dtime
        self.time_right = time + self.dtime

        Lu = self.t_fac * self.prodF(u, u, time, initial=True)
        Lu, info = gmres(A=self.Bop1, b=Lu, x0=Lu, M=self.PC1, rtol=1e-10)
        u1 = u + (0.5 * self.dtime) * Lu

        Lu1 = Lu + self.getW(u1, self.time_mid)
        Lu1, info = gmres(A=self.Bop2, b=Lu1, x0=Lu1, M=self.PC2, rtol=1e-10)
        u1 = u + self.dtime * Lu1
        return u1

    def prop(self, u_init, t_list, callback=None):
        u0 = u_init.copy()
        if callback is not None:
            callback(u0, time=t_list[0])
        for i in tqdm(range(1, len(t_list))):
            u1 = self.prop1_etdrk2(
                u=u0, time=t_list[i - 1], dtime=t_list[i] - t_list[i - 1]
            )
            if callback is not None:
                callback(u1, time=t_list[i])
            u0 = u1
        return u1
