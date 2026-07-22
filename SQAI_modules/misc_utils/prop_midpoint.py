import numpy as np
from tqdm.notebook import tqdm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, splu, gmres

from .prop_wrapper import prop_wrapper


class prop_midpoint(prop_wrapper):
    def __init__(self, imaginary, dtime, TDHF, Laser=None, midpoint_type=0):
        super().__init__(imaginary=imaginary, dtime=dtime, TDHF=TDHF, Laser=Laser)
        self.midpoint_type = midpoint_type
        self.setup()

    def setup(self):
        self.Bop1 = LinearOperator(
            shape=(self.L_size, self.L_size),
            matvec=lambda u: self.Lxpy(u, u, self.time_left, Ffac=-0.25, initial=False),
        )
        self.Bop2 = LinearOperator(
            shape=(self.L_size, self.L_size),
            matvec=lambda u: self.Lxpy(u, u, self.time_mid, Ffac=-0.50, initial=False),
        )
        if self.midpoint_type == 1:
            self.Bop2a = LinearOperator(
                shape=(self.L_size, self.L_size),
                matvec=lambda u: self.Lxpy(
                    u, u, self.time_14, Ffac=-0.25, initial=False
                ),
            )
            self.Bop2b = LinearOperator(
                shape=(self.L_size, self.L_size),
                matvec=lambda u: self.Lxpy(
                    u, u, self.time_34, Ffac=-0.25, initial=False
                ),
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

    def prop1_midpoint(self, u, time, dtime=None):
        self.time_left = time
        self.time_mid = time + 0.5 * self.dtime
        self.time_right = time + self.dtime
        self.time_14 = time + 0.25 * self.dtime
        self.time_34 = time + 0.75 * self.dtime
        # debug
        #        print('tL', self.time_left)
        #        print('tM', self.time_mid)
        #        print('tR', self.time_right)
        # debug
        u1 = self.Lxpy(u, u, self.time_left, Ffac=0.25, initial=True)
        u1, info = gmres(A=self.Bop1, b=u1, x0=u1, M=self.PC1, rtol=1e-10)

        if self.midpoint_type == 0:
            u1 = self.Lxpy(u, u1, self.time_mid, Ffac=0.50, initial=True)
            u1, info = gmres(A=self.Bop2, b=u1, x0=u1, M=self.PC2, rtol=1e-10)
        elif self.midpoint_type == 1:
            u_mid = u1.copy()
            u1 = self.Lxpy(u, u_mid, self.time_14, Ffac=0.25, initial=True)
            u1, info = gmres(A=self.Bop2a, b=u1, x0=u1, M=self.PC1, rtol=1e-10)
            u1 = self.Lxpy(u1, u_mid, self.time_34, Ffac=0.25, initial=True)
            u1, info = gmres(A=self.Bop2b, b=u1, x0=u1, M=self.PC1, rtol=1e-10)
        else:
            raise Exception("prop1_midpoint: bad midpoint_type.")

        return u1

    def prop(self, u_init, t_list, callback=None):
        u0 = u_init.copy()
        if callback is not None:
            callback(u0, time=t_list[0])
        for i in tqdm(range(1, len(t_list))):
            u1 = self.prop1_midpoint(
                u=u0, time=t_list[i - 1], dtime=t_list[i] - t_list[i - 1]
            )
            if callback is not None:
                callback(u1, time=t_list[i])
            u0 = u1
        return u1

    def prop_new(self, u_init, t_list, callback=None):
        u0 = u_init.copy()
        self.u_half_before = u0.copy()
        self.u_half_after = u0.copy()

        if callback is not None:
            callback(u0, time=t_list[0])

        for i in tqdm(range(1, len(t_list))):
            u1 = self.prop1(u=u0, time=t_list[i - 1], dtime=t_list[i] - t_list[i - 1])
            if callback is not None:
                callback(u1, time=t_list[i])
            u0 = u1
        return u1

    def prop1(self, u, time, dtime=None):
        self.u_half_after[:] = self.prop1_mp2(
            u_left=self.u_half_before, u_mid=u, time_mid=time
        )
        u1 = self.prop1_mp2(
            u_left=u, u_mid=self.u_half_after, time_mid=time + 0.5 * self.dtime
        )
        self.u_half_before[:] = self.u_half_after[:]
        return u1

    def prop1_mp2(self, u_left, u_mid, time_mid, dtime=None):
        self.time_mid = time_mid
        ut = self.Lxpy(u_left, u_mid, time_mid, Ffac=0.5, initial=True)
        u_right, info = gmres(A=self.Bop2, b=ut, x0=ut, M=self.PC2, rtol=1e-10)
        return u_right
