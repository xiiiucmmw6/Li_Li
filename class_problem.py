from dolfin import *

class West(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.0 + DOLFIN_EPS and on_boundary

class East(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 1.0 - DOLFIN_EPS and on_boundary

class Problem:

    def store(self, t, u):

        assign(self.c_e_plot, u.sub(0))
        assign(self.phi_plot, u.sub(1))

        self.c_e_pvd << (self.c_e_plot, t)
        self.phi_pvd << (self.phi_plot, t)

    def __init__(self):

        mesh = UnitIntervalMesh(100)

        P1 = FiniteElement("CG", mesh.ufl_cell(), 1); ME = MixedElement([P1, P1])

        self.W = FunctionSpace(mesh, ME)
        self.V = FunctionSpace(mesh, P1)

        self.u_1 = Function(self.W)
        self.u_0 = Function(self.W)

        (self.c_e_1, self.phi_1) = split(self.u_1)
        (self.c_e_0, self.phi_0) = split(self.u_0)

        self.c_e_plot = Function(self.V)
        self.phi_plot = Function(self.V)

        self.c_e_pvd = File('results/c_e.pvd')
        self.phi_pvd = File('results/phi.pvd')


        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1); boundaries.set_all(0)

        west = West()
        east = East()

        west.mark(boundaries, 1)
        east.mark(boundaries, 2)

        self.dx = Measure("dx", domain=mesh)
        self.ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

        self.c_e_ini = Constant(1000)

        self.k_eff = Constant(1)
        self.k_D_eff = Constant(1)
        self.D_e_eff = Constant(0.01)
        self.i_app = Constant(1)
        self.A = Constant(1)
        self.h = Expression('value',degree=0,value=0.1)
        self.eps_e = Constant(0.5)
        self.F = Constant(1)
        self.R = Constant(1)
        self.T = Constant(1)
        self.alpha = Constant(0.5)
        self.k_0 =Constant(1)

        (c_e, phi) = TestFunction(self.W)

        j_Li = self.k_0 * self.c_e_0 ** self.alpha * exp((1 - self.alpha) * (self.F/(self.R*self.T)) * self.phi_0) \
             - self.k_0 * self.c_e_0 ** self.alpha * exp((0 - self.alpha) * (self.F/(self.R*self.T)) * self.phi_0)

        F_c_e = self.c_e_0 * c_e * self.dx - self.c_e_ini * c_e * self.dx

        F_phi = inner(self.k_eff * grad(self.phi_0) + self.k_D_eff * (1. / self.c_e_0) * grad(self.c_e_0), grad(phi)) * self.dx \
              - (self.i_app / self.A) * phi * self.ds(1) \
              + (self.i_app / self.A) * phi * self.ds(2)

        self.F_0 = F_c_e + F_phi; self.J_0 = derivative(self.F_0, self.u_0)

        assign(self.u_0.sub(0), interpolate(self.c_e_ini,self.V))

        problem = NonlinearVariationalProblem(self.F_0, self.u_0, J=self.J_0); solver = NonlinearVariationalSolver(problem); prm = solver.parameters

        prm["newton_solver"]["absolute_tolerance"] = 1e-6
        prm["newton_solver"]["relative_tolerance"] = 1e-6
        prm["newton_solver"]["maximum_iterations"] = 100; prm["newton_solver"]["relaxation_parameter"] = 1.0

        solver.solve()

        self.store(0., self.u_0)

        j_Li = self.k_0 * self.c_e_1 ** self.alpha * exp((1 - self.alpha) * (self.F/(self.R*self.T)) * self.phi_1) \
             - self.k_0 * self.c_e_1 ** self.alpha * exp((0 - self.alpha) * (self.F/(self.R*self.T)) * self.phi_1)

        F_c_e = self.eps_e * ((self.c_e_1 - self.c_e_0) / self.h) * c_e * self.dx + inner(self.D_e_eff * grad(self.c_e_1), grad(c_e)) * self.dx \
              - (self.i_app / self.A) * c_e * self.ds(1) \
              + (self.i_app / self.A) * c_e * self.ds(2)

        F_phi = inner(self.k_eff * grad(self.phi_1) + self.k_D_eff * (1. / self.c_e_1) * grad(self.c_e_1), grad(phi)) * self.dx \
              - j_Li * phi * self.ds(1) \
              + j_Li * phi * self.ds(2)

        self.F = F_c_e + F_phi; self.J = derivative(self.F, self.u_1)

        self.problem = NonlinearVariationalProblem(self.F, self.u_1, J=self.J); self.solver = NonlinearVariationalSolver(self.problem); prm = solver.parameters

        prm["newton_solver"]["absolute_tolerance"] = 1e-6
        prm["newton_solver"]["relative_tolerance"] = 1e-6
        prm["newton_solver"]["maximum_iterations"] = 100; prm["newton_solver"]["relaxation_parameter"] = 1.0

        self.t = 0

        assign(self.u_1, self.u_0)

        while self.t < 10.0:
            self.tstep(0.1)

    def tstep(self, h):

        self.h.value = h; self.t = self.t + h

        self.solver.solve()

        assign(self.u_0, self.u_1)

        self.store(self.t, self.u_1)


if __name__ == '__main__':

    problem = Problem()
