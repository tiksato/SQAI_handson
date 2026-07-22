import numpy as np



##############################
class propagator:
    def __init__(self, solver, dtime=0.1):

        self.solver = solver
        self.dtime = dtime
        self.normalize_at_each_step = True

    def prop1(self, u0):
        return u0.copy()

    def imaginary_time_relaxation(
        self, u_init, maxcyc=10000, thresh=1e-15, print_every=1
    ):
        u_now = u_init.copy()
        energy_previous = 1e10

        for cyc in range(maxcyc):
            energy = self.solver.calc_energy(u_now)
            energy_diff = energy - energy_previous

            if cyc % print_every == 0:
                print(f"{cyc:10d} {energy:15.10f} {energy_diff:12.5e}")
            if np.abs(energy_diff) < thresh:
                print(f"{cyc:10d} {energy:15.10f} {energy_diff:12.5e}")
                break

            energy_previous = energy
            u_now = self.prop1(u_now)

            self.solver.orthonormalize(u_now)

        u_opt = u_now

        return u_opt

    def real_time_propagation(self, u_init, maxcyc=1001, save_every=1, print_every=100):
        u_now = u_init.copy()
        u_list = []

        for cyc in range(maxcyc):

            if cyc % save_every == 0:
                u_list.append(u_now.copy())

            if cyc % print_every == 0:
                norm = np.linalg.norm(u_now)
                energy = self.solver.calc_energy(u_now)
                print(f"{cyc:10d} {energy:15.10f} {norm:15.10f}")

            u_now = self.prop1(u_now)

        return u_list


##############################
class propagator_rk1(propagator):
    def __init__(self, solver, dtime=0.1):
        super().__init__(solver, dtime)

    def prop1(self, u0):
        u1 = u0 + self.dtime * self.solver.getRHS(u0)
        return u1


##############################
class propagator_rk2(propagator):
    def __init__(self, solver, dtime=0.1):
        super().__init__(solver, dtime)

    def prop1(self, u0):
        k1 = self.solver.getRHS(u0)
        u_tmp = u0 + (self.dtime / 2) * k1
        if self.imaginary and self.normalize_at_each_step:
            self.solver.orthonormalize(u_tmp)
        k2 = self.solver.getRHS(u_tmp)
        u1 = u0 + self.dtime * k2
        return u1


##############################
class propagator_rk4(propagator):
    def __init__(self, solver, dtime=0.1):
        super().__init__(solver, dtime)

    def prop1(self, u0):
        k1 = self.solver.getRHS(u0)
        u1 = u0 + (self.dtime / 6) * k1

        u_tmp = u0 + (self.dtime / 2) * k1
        if self.normalize_at_each_step:
            self.solver.orthonormalize(u_tmp)
        k2 = self.solver.getRHS(u_tmp)
        u1 += (self.dtime / 3) * k2

        u_tmp = u0 + (self.dtime / 2) * k2
        if self.normalize_at_each_step:
            self.solver.orthonormalize(u_tmp)
        k3 = self.solver.getRHS(u_tmp)
        u1 += (self.dtime / 3) * k3

        u_tmp = u0 + self.dtime * k3
        if self.normalize_at_each_step:
            self.solver.orthonormalize(u_tmp)
        k4 = self.solver.getRHS(u_tmp)
        u1 += (self.dtime / 6) * k4
        return u1
