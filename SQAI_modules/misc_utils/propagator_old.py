from scipy.sparse.linalg import gmres, LinearOperator


class propagator:
    def __init__(self, dtime, dudt):

        self.time = 0.0
        self.dtime = dtime
        self.dudt = dudt

    def steps(self, u_init, num_steps, stop_check=None, callback=None):
        u = u_init.copy()
        u_previous = u.copy()
        self.time = 0.0
        if callback is not None:
            callback(self.time, u)

        for i_time in range(1, num_steps):
            u = self.step1(self.time, u)
            if callback is not None:
                callback(self.time, u)
            if stop_check is not None:
                if stop_check(u, u_previous):
                    break
            self.time += self.dtime
            u_previous = u.copy()
        return u


class etdrk1_pade11(propagator):
    def __init__(self, dtime, dudt, phi1_denominator=None, phi1_preconditioner=None):

        super().__init__(dtime, dudt)
        self.phi1_denominator = phi1_denominator
        self.phi1_preconditioner = phi1_preconditioner

    def step1(self, time, u):
        u1 = self.dudt(time, u)
        # print('u1.shape = ', u1.shape)
        if self.phi1_denominator is not None:
            A = LinearOperator(
                shape=(len(u), len(u)),
                matvec=lambda u: self.phi1_denominator(
                    time,
                    self.dtime,
                    u,
                ),
            )
            # u1, info = gmres(A = A,
            #                 b = u1,
            #                 x0 = u, rtol=1e-05, atol=0.0,
            #                 M = self.phi1_preconditioner)
            u1, info = gmres(A=A, b=u1, x0=u, rtol=1e-05, atol=0.0)
        # print('u1.shape = ', u1.shape)
        u[:] += self.dtime * u1
        return u
