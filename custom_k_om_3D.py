"""Equations related to Navier Stokes Equations
"""

from sympy import Symbol, Function, Number, log, Abs, simplify, pi

from modulus.eq.pde import PDE
from modulus.node import Node


class kOmegaInit(PDE):
    def __init__(self, nu=1, rho=1):
        # set params
        nu = Number(nu)
        rho = Number(rho)

        # coordinates
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")  # Adding z coordinate

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}  # Adding z to input variables

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        w = Function("w")(*input_variables)  # Adding w component
        p = Function("p")(*input_variables)
        k = Function("k")(*input_variables)
        om_plus = Function("om_plus")(*input_variables)

        # flow initialization
        C_mu = 0.09
        u_avg = 15  # Approx average velocity
        Re_d = (
            u_avg * 1 / nu
        )  # Reynolds number based on centerline and channel hydraulic dia
        l = 0.038 * 2  # Approx turbulent length scale
        I = 0.16 * Re_d ** (
            -1 / 8
        )  # Turbulent intensity for a fully developed pipe flow

        u_init = 0
        v_init = u_avg
        w_init = 0  # Setting initial value for w component
        p_init = pi / 2
        k_init = 1.5 * (u_avg * I) ** 2
        ep_init = (C_mu ** (3 / 4)) * (k_init ** (3 / 2)) / l
        om_plus_init = ep_init / C_mu / k_init * nu  # Solving for om_plus

        # set equations
        self.equations = {}
        self.equations["u_init"] = u - u_init
        self.equations["v_init"] = v - v_init
        self.equations["w_init"] = w - w_init  # Adding w initialization equation
        self.equations["p_init"] = p - p_init
        self.equations["k_init"] = k - k_init
        self.equations["om_plus_init"] = om_plus - om_plus_init

class TransientkOmega(PDE):
    def __init__(self, nu=1, rho=1):
        # set params
        nu = Number(nu)
        rho = Number(rho)

        # coordinates
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")
        t = Symbol("t")  # Adding time variable

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}  # Adding t to input variables

        # velocity components
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        w = Function("w")(*input_variables)
        p = Function("p")(*input_variables)
        k = Function("k")(*input_variables)
        om_plus = Function("om_plus")(*input_variables)

        # Model constants
        sig = 0.5
        sig_star = 0.5
        C_mu = 0.09
        E = 9.793
        beta = 3 / 40
        alpha = 5 / 9
        beta_star = 9 / 100

        # Turbulent Viscosity
        nu_t = k * nu / (om_plus + 1e-4)

        # Turbulent Production Term
        P_k = nu_t * (
            2 * (u.diff(x)) ** 2
            + 2 * (v.diff(y)) ** 2
            + 2 * (w.diff(z)) ** 2
            + (u.diff(y) + v.diff(x)) ** 2
            + (u.diff(z) + w.diff(x)) ** 2
            + (v.diff(z) + w.diff(y)) ** 2
        )

        # set equations
        self.equations = {}
        self.equations["continuity"] = simplify(u.diff(x) + v.diff(y) + w.diff(z))
        self.equations["momentum_x"] = simplify(
            rho * u.diff(t)
            + u * u.diff(x)
            + v * u.diff(y)
            + w * u.diff(z)
            + p.diff(x)
            - ((nu + nu_t) * u.diff(x)).diff(x)
            - ((nu + nu_t) * u.diff(y)).diff(y)
            - ((nu + nu_t) * u.diff(z)).diff(z)
        )
        self.equations["momentum_y"] = simplify(
            rho * v.diff(t)
            + u * v.diff(x)
            + v * v.diff(y)
            + w * v.diff(z)
            + p.diff(y)
            - ((nu + nu_t) * v.diff(x)).diff(x)
            - ((nu + nu_t) * v.diff(y)).diff(y)
            - ((nu + nu_t) * v.diff(z)).diff(z)
        )
        self.equations["momentum_z"] = simplify(
            rho * w.diff(t)
            + u * w.diff(x)
            + v * w.diff(y)
            + w * w.diff(z)
            + p.diff(z)
            - ((nu + nu_t) * w.diff(x)).diff(x)
            - ((nu + nu_t) * w.diff(y)).diff(y)
            - ((nu + nu_t) * w.diff(z)).diff(z)
        )
        self.equations["k_equation"] = simplify(
            k.diff(t)
            + u * k.diff(x)
            + v * k.diff(y)
            + w * k.diff(z)
            - ((nu + nu_t * sig_star) * k.diff(x)).diff(x)
            - ((nu + nu_t * sig_star) * k.diff(y)).diff(y)
            - ((nu + nu_t * sig_star) * k.diff(z)).diff(z)
            - P_k
            + beta_star * k * om_plus / nu
        )
        self.equations["om_plus_equation"] = simplify(
            om_plus.diff(t)
            + u * om_plus.diff(x) / nu
            + v * om_plus.diff(y) / nu
            + w * om_plus.diff(z) / nu
            - ((nu + nu_t * sig) * om_plus.diff(x)).diff(x) / nu
            - ((nu + nu_t * sig) * om_plus.diff(y)).diff(y) / nu
            - ((nu + nu_t * sig) * om_plus.diff(z)).diff(z) / nu
            - alpha
            * (
                2 * (u.diff(x)) ** 2
                + 2 * (v.diff(y)) ** 2
                + 2 * (w.diff(z)) ** 2
                + (u.diff(y) + v.diff(x)) ** 2
                + (u.diff(z) + w.diff(x)) ** 2
                + (v.diff(z) + w.diff(y)) ** 2
            )
            + beta * om_plus * om_plus / nu / nu
        )

class kOmega(PDE):
    def __init__(self, nu=1, rho=1):
        # set params
        nu = Number(nu)
        rho = Number(rho)

        # coordinates
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")  # Adding z coordinate

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}  # Adding z to input variables

        # velocity components
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        w = Function("w")(*input_variables)  # Adding w component
        p = Function("p")(*input_variables)
        k = Function("k")(*input_variables)
        om_plus = Function("om_plus")(*input_variables)  # Solving for om_plus

        # Model constants
        sig = 0.5
        sig_star = 0.5
        C_mu = 0.09
        E = 9.793
        beta = 3 / 40
        alpha = 5 / 9
        beta_star = 9 / 100

        # Turbulent Viscosity
        nu_t = k * nu / (om_plus + 1e-4)

        # Turbulent Production Term
        P_k = nu_t * (
            2 * (u.diff(x)) ** 2
            + 2 * (v.diff(y)) ** 2
            + 2 * (w.diff(z)) ** 2  # Adding w component
            + (u.diff(y) + v.diff(x)) ** 2
            + (u.diff(z) + w.diff(x)) ** 2  # Adding w component
            + (v.diff(z) + w.diff(y)) ** 2  # Adding w component
        )

        # set equations
        self.equations = {}
        self.equations["continuity"] = simplify(u.diff(x) + v.diff(y) + w.diff(z))  # Adding w component
        self.equations["momentum_x"] = simplify(
            u * u.diff(x)
            + v * u.diff(y)
            + w * u.diff(z)  # Adding w component
            + p.diff(x)
            - ((nu + nu_t) * u.diff(x)).diff(x)
            - ((nu + nu_t) * u.diff(y)).diff(y)
            - ((nu + nu_t) * u.diff(z)).diff(z)  # Adding w component
        )
        self.equations["momentum_y"] = simplify(
            u * v.diff(x)
            + v * v.diff(y)
            + w * v.diff(z)  # Adding w component
            + p.diff(y)
            - ((nu + nu_t) * v.diff(x)).diff(x)
            - ((nu + nu_t) * v.diff(y)).diff(y)
            - ((nu + nu_t) * v.diff(z)).diff(z)  # Adding w component
        )
        self.equations["momentum_z"] = simplify(
            u * w.diff(x)  # Adding w component
            + v * w.diff(y)  # Adding w component
            + w * w.diff(z)
            + p.diff(z)
            - ((nu + nu_t) * w.diff(x)).diff(x)  # Adding w component
            - ((nu + nu_t) * w.diff(y)).diff(y)  # Adding w component
            - ((nu + nu_t) * w.diff(z)).diff(z)
        )
        self.equations["k_equation"] = simplify(
            u * k.diff(x)
            + v * k.diff(y)
            + w * k.diff(z)  # Adding w component
            - ((nu + nu_t * sig_star) * k.diff(x)).diff(x)
            - ((nu + nu_t * sig_star) * k.diff(y)).diff(y)
            - ((nu + nu_t * sig_star) * k.diff(z)).diff(z)  # Adding w component
            - P_k
            + beta_star * k * om_plus / nu
        )
        self.equations["om_plus_equation"] = simplify(
            u * om_plus.diff(x) / nu
            + v * om_plus.diff(y) / nu
            + w * om_plus.diff(z) / nu  # Adding w component
            - ((nu + nu_t * sig) * om_plus.diff(x)).diff(x) / nu
            - ((nu + nu_t * sig) * om_plus.diff(y)).diff(y) / nu
            - ((nu + nu_t * sig) * om_plus.diff(z)).diff(z) / nu  # Adding w component
            - alpha
            * (
                2 * (u.diff(x)) ** 2
                + 2 * (v.diff(y)) ** 2
                + 2 * (w.diff(z)) ** 2  # Adding w component
                + (u.diff(y) + v.diff(x)) ** 2
                + (u.diff(z) + w.diff(x)) ** 2  # Adding w component
                + (v.diff(z) + w.diff(y)) ** 2  # Adding w component
            )
            + beta * om_plus * om_plus / nu / nu
        )

class kOmegaStdWF(PDE):
    def __init__(self, nu=1, rho=1):
        # set params
        nu = Number(nu)
        rho = Number(rho)

        # coordinates
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")  # Adding z coordinate

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}  # Adding z to input variables

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        w = Function("w")(*input_variables)  # Adding w component
        k = Function("k")(*input_variables)
        om_plus = Function("om_plus")(*input_variables)

        # normals
        normal_x = -1 * Symbol("normal_x")  # Multiply by -1 to flip the direction of normal
        normal_y = -1 * Symbol("normal_y")  # Multiply by -1 to flip the direction of normal
        normal_z = -1 * Symbol("normal_z")  # Multiply by -1 to flip the direction of normal

        # wall distance
        normal_distance = Function("normal_distance")(*input_variables)

        # Model constants
        C_mu = 0.09
        E = 9.793
        C_k = -0.36
        B_k = 8.15
        karman_constant = 0.4187
        beta_star = 9 / 100

        # Turbulent Viscosity
        nu_t = k * nu / (om_plus + 1e-4)

        u_tau = Function("u_tau")(*input_variables)
        y_plus = u_tau * normal_distance / nu
        u_plus = log(Abs(E * y_plus) + 1e-3) / karman_constant

        om_plus_true = (
            (k ** 0.5) / (beta_star ** 0.25) / karman_constant / normal_distance
        ) * nu

        u_parallel_to_wall = [
            u - (u * normal_x + v * normal_y + w * normal_z) * normal_x,
            v - (u * normal_x + v * normal_y + w * normal_z) * normal_y,
            w - (u * normal_x + v * normal_y + w * normal_z) * normal_z,
        ]
        du_parallel_to_wall_dx = [
            u.diff(x) - (u.diff(x) * normal_x + v.diff(x) * normal_y + w.diff(x) * normal_z) * normal_x,
            v.diff(x) - (u.diff(x) * normal_x + v.diff(x) * normal_y + w.diff(x) * normal_z) * normal_y,
            w.diff(x) - (u.diff(x) * normal_x + v.diff(x) * normal_y + w.diff(x) * normal_z) * normal_z,
        ]
        du_parallel_to_wall_dy = [
            u.diff(y) - (u.diff(y) * normal_x + v.diff(y) * normal_y + w.diff(y) * normal_z) * normal_x,
            v.diff(y) - (u.diff(y) * normal_x + v.diff(y) * normal_y + w.diff(y) * normal_z) * normal_y,
            w.diff(y) - (u.diff(y) * normal_x + v.diff(y) * normal_y + w.diff(y) * normal_z) * normal_z,
        ]
        du_parallel_to_wall_dz = [
            u.diff(z) - (u.diff(z) * normal_x + v.diff(z) * normal_y + w.diff(z) * normal_z) * normal_x,
            v.diff(z) - (u.diff(z) * normal_x + v.diff(z) * normal_y + w.diff(z) * normal_z) * normal_y,
            w.diff(z) - (u.diff(z) * normal_x + v.diff(z) * normal_y + w.diff(z) * normal_z) * normal_z,
        ]

        du_dsdf = [
            du_parallel_to_wall_dx[0] * normal_x + du_parallel_to_wall_dy[0] * normal_y + du_parallel_to_wall_dz[0] * normal_z,
            du_parallel_to_wall_dx[1] * normal_x + du_parallel_to_wall_dy[1] * normal_y + du_parallel_to_wall_dz[1] * normal_z,
            du_parallel_to_wall_dx[2] * normal_x + du_parallel_to_wall_dy[2] * normal_y + du_parallel_to_wall_dz[2] * normal_z,
        ]

        wall_shear_stress_true_x = (
            u_tau
            * u_parallel_to_wall[0]
            * karman_constant
            / log(Abs(E * y_plus) + 1e-3)
        )
        wall_shear_stress_true_y = (
            u_tau
            * u_parallel_to_wall[1]
            * karman_constant
            / log(Abs(E * y_plus) + 1e-3)
        )
        wall_shear_stress_true_z = (
            u_tau
            * u_parallel_to_wall[2]
            * karman_constant
            / log(Abs(E * y_plus) + 1e-3)
        )

        wall_shear_stress_x = (nu + nu_t) * du_dsdf[0]
        wall_shear_stress_y = (nu + nu_t) * du_dsdf[1]
        wall_shear_stress_z = (nu + nu_t) * du_dsdf[2]

        u_normal_to_wall = u * normal_x + v * normal_y + w * normal_z
        u_normal_to_wall_true = 0

        u_parallel_to_wall_mag = (
            u_parallel_to_wall[0] ** 2 + u_parallel_to_wall[1] ** 2 + u_parallel_to_wall[2] ** 2
        ) ** 0.5
        u_parallel_to_wall_true = u_plus * u_tau

        # k_normal_gradient = normal_x*k.diff(x) + normal_y*k.diff(y)
        # k_normal_gradient_true = 0
        k_true = u_tau ** 2 / C_mu ** 0.5

        # set equations
        self.equations = {}
        self.equations["velocity_wall_normal_wf"] = (
            u_normal_to_wall - u_normal_to_wall_true
        )
        self.equations["velocity_wall_parallel_wf"] = (
            u_parallel_to_wall_mag - u_parallel_to_wall_true
        )
        self.equations["k_wf"] = k - k_true
        self.equations["om_plus_wf"] = om_plus - om_plus_true
        self.equations["wall_shear_stress_x_wf"] = (
            wall_shear_stress_x - wall_shear_stress_true_x
        )
        self.equations["wall_shear_stress_y_wf"] = (
            wall_shear_stress_y - wall_shear_stress_true_y
        )
        self.equations["wall_shear_stress_z_wf"] = (
            wall_shear_stress_z - wall_shear_stress_true_z
        )


