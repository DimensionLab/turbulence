"""Equations related to Navier Stokes Equations
"""

from sympy import Symbol, Function, Number, log, Abs, simplify

from modulus.eq.pde import PDE
from modulus.node import Node


class kEpsilonInit(PDE):
    def __init__(self, nu=1, rho=1):
        # set params
        nu = Number(nu)
        rho = Number(rho)

        # coordinates
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        w = Function("w")(*input_variables)
        p = Function("p")(*input_variables)
        k = Function("k")(*input_variables)
        ep = Function("ep")(*input_variables)

        # flow initialization
        C_mu = 0.09
        u_avg = 8  # Approx average velocity
        Re_d = (
            u_avg * 1 / nu
        )  # Reynolds number based on centerline and channel hydraulic dia
        l = 0.07 * 2  # Approx turbulent length scale 
        """ In fully developed pipe flow, the turbulence length scale can be estimated from the hydraulic diameter (( D_h )) of the pipe.
         A common approximation is to set ( l ) to be around 3.8% of the hydraulic diameter
         it should be higher in wind tunnel scenarios, lets set it to 7% here
         https://www.cfd-online.com/Wiki/Turbulence_length_scale """
        I = 0.16 * Re_d ** (
            -1 / 8
        )  # Turbulent intensity for a fully developed pipe flow

        u_init = 0
        v_init = u_avg
        w_init = 0
        p_init = 0
        k_init = 1.5 * (u_avg * I) ** 2
        ep_init = (C_mu ** (3 / 4)) * (k_init ** (3 / 2)) / l

        # set equations
        self.equations = {}
        self.equations["u_init"] = u - u_init
        self.equations["v_init"] = v - v_init
        self.equations["w_init"] = w - w_init
        self.equations["p_init"] = p - p_init
        self.equations["k_init"] = k - k_init
        self.equations["ep_init"] = ep - ep_init


class kEpsilon(PDE):
    def __init__(self, nu=1, rho=1):
        # set params
        nu = Number(nu)
        rho = Number(rho)

        # coordinates
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")  # Add z coordinate
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}

        # velocity components
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        w = Function("w")(*input_variables)  # Add w velocity component
        p = Function("p")(*input_variables)
        k = Function("k")(*input_variables)
        ep = Function("ep")(*input_variables)

        # Model constants
        sig_k = Number(1.0)
        sig_ep = Number(1.3)
        C_ep1 = Number(1.44)
        C_ep2 = Number(1.92)
        C_mu = Number(0.09)
        E = Number(9.793)

        # Turbulent Viscosity
        nu_t = C_mu * (k ** 2) / (ep + 1e-4)

        # Turbulent Production Term
        P_k = nu_t * (
            2 * (u.diff(x)) ** 2
            + 2 * (v.diff(y)) ** 2
            + 2 * (w.diff(z)) ** 2  # Add w term
            + (u.diff(y) + v.diff(x)) ** 2
            + (u.diff(z) + w.diff(x)) ** 2  # Add w term
            + (v.diff(z) + w.diff(y)) ** 2  # Add w term
        )

        # set equations
        self.equations = {}
        self.equations["continuity"] = simplify(u.diff(x) + v.diff(y) + w.diff(z))  # Modify for 3D
        self.equations["momentum_x"] = simplify(
            u * u.diff(x)
            + v * u.diff(y)
            + w * u.diff(z)  # Add w term
            + p.diff(x)
            - ((nu + nu_t) * u.diff(x)).diff(x)
            - ((nu + nu_t) * u.diff(y)).diff(y)
            - ((nu + nu_t) * u.diff(z)).diff(z)  # Add w term
        )
        self.equations["momentum_y"] = simplify(
            u * v.diff(x)
            + v * v.diff(y)
            + w * v.diff(z)  # Add w term
            + p.diff(y)
            - ((nu + nu_t) * v.diff(x)).diff(x)
            - ((nu + nu_t) * v.diff(y)).diff(y)
            - ((nu + nu_t) * v.diff(z)).diff(z)  # Add w term
        )
        self.equations["momentum_z"] = simplify(  # Add momentum equation for z
            u * w.diff(x)
            + v * w.diff(y)
            + w * w.diff(z)
            + p.diff(z)
            - ((nu + nu_t) * w.diff(x)).diff(x)
            - ((nu + nu_t) * w.diff(y)).diff(y)
            - ((nu + nu_t) * w.diff(z)).diff(z)
        )
        self.equations["k_equation"] = (
            u * k.diff(x)
            + v * k.diff(y)
            + w * k.diff(z)  # Add w term
            - ((nu + nu_t / sig_k) * k.diff(x)).diff(x)
            - ((nu + nu_t / sig_k) * k.diff(y)).diff(y)
            - ((nu + nu_t / sig_k) * k.diff(z)).diff(z)  # Add w term
            - P_k
            + ep
        )
        self.equations["ep_equation"] = (
            u * ep.diff(x)
            + v * ep.diff(y)
            + w * ep.diff(z)  # Add w term
            - ((nu + nu_t / sig_ep) * ep.diff(x)).diff(x)
            - ((nu + nu_t / sig_ep) * ep.diff(y)).diff(y)
            - ((nu + nu_t / sig_ep) * ep.diff(z)).diff(z)  # Add w term
            - (C_ep1 * P_k - C_ep2 * ep) * ep / (k + 1e-3)
        )


class kEpsilonLSWF(PDE):
    def __init__(self, nu=1, rho=1, y_plus=30):
        # set params
        nu = Number(nu)
        rho = Number(rho)

        # wall distance
        normal_distance = Number(y_plus)

        # coordinates
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}

        # velocity components
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        w = Function("w")(*input_variables)  # w component for the third dimension
        k = Function("k")(*input_variables)
        ep = Function("ep")(*input_variables)

        # normals
        normal_x = -1 * Symbol("normal_x")  # Flip the direction of normal
        normal_y = -1 * Symbol("normal_y")  # Flip the direction of normal
        normal_z = -1 * Symbol("normal_z")  # Flip the direction of normal


        # Model constants
        C_mu = 0.09
        E = 9.793
        C_k = -0.36
        B_k = 8.15
        karman_constant = 0.4187

        # Turbulent Viscosity
        nu_t = C_mu * (k ** 2) / (ep + 1e-4)

        u_tau = (C_mu ** 0.25) * (k ** 0.5)
        y_plus = u_tau * normal_distance / nu
        u_plus = log(Abs(E * y_plus)) / karman_constant

        ep_true = (C_mu ** (3 / 4)) * (k ** (3 / 2)) / karman_constant / normal_distance

        u_parallel_to_wall = [
            u - (u * normal_x + v * normal_y + w * normal_z) * normal_x,
            v - (u * normal_x + v * normal_y + w * normal_z) * normal_y,
            w - (u * normal_x + v * normal_y + w * normal_z) * normal_z,  # Added third component
        ]
        # Added derivatives for the third dimension
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
        # Add derivatives with respect to z
        du_parallel_to_wall_dz = [
            u.diff(z) - (u.diff(z) * normal_x + v.diff(z) * normal_y + w.diff(z) * normal_z) * normal_x,
            v.diff(z) - (u.diff(z) * normal_x + v.diff(z) * normal_y + w.diff(z) * normal_z) * normal_y,
            w.diff(z) - (u.diff(z) * normal_x + v.diff(z) * normal_y + w.diff(z) * normal_z) * normal_z,
        ]

        # Correct the calculation of the derivative in the direction of the surface (du/ds)
        du_dsdf = [
            du_parallel_to_wall_dx[0] * normal_x + du_parallel_to_wall_dy[0] * normal_y + du_parallel_to_wall_dz[0] * normal_z,
            du_parallel_to_wall_dx[1] * normal_x + du_parallel_to_wall_dy[1] * normal_y + du_parallel_to_wall_dz[1] * normal_z,
            du_parallel_to_wall_dx[2] * normal_x + du_parallel_to_wall_dy[2] * normal_y + du_parallel_to_wall_dz[2] * normal_z,
        ]
        # Update wall shear stresses to include the w component
        wall_shear_stress_true_x = (
            u_tau * u_parallel_to_wall[0] * karman_constant / log(Abs(E * y_plus))
        )
        wall_shear_stress_true_y = (
            u_tau * u_parallel_to_wall[1] * karman_constant / log(Abs(E * y_plus))
        )
        # Introducing wall shear stress in the z direction for completeness
        wall_shear_stress_true_z = (
            u_tau * u_parallel_to_wall[2] * karman_constant / log(Abs(E * y_plus))
        )

        # Calculating the wall shear stress in all three directions
        wall_shear_stress_x = (nu + nu_t) * du_dsdf[0]
        wall_shear_stress_y = (nu + nu_t) * du_dsdf[1]
        wall_shear_stress_z = (nu + nu_t) * du_dsdf[2]

        # Velocity normal to the wall
        u_normal_to_wall = u * normal_x + v * normal_y + w * normal_z
        u_normal_to_wall_true = 0

        # Magnitude of the velocity parallel to the wall, now including the w component
        u_parallel_to_wall_mag = (
            u_parallel_to_wall[0] ** 2 + u_parallel_to_wall[1] ** 2 + u_parallel_to_wall[2] ** 2
        ) ** 0.5
        u_parallel_to_wall_true = u_plus * u_tau

        # Gradient of k normal to the wall, now accounting for all three spatial directions
        # k_normal_gradient = normal_x * k.diff(x) + normal_y * k.diff(y) + normal_z * k.diff(z)
        # k_normal_gradient_true = 0
        k_true = u_tau ** 2 / C_mu ** 0.5

        # Set equations with 3D considerations
        self.equations = {}
        self.equations["velocity_wall_normal_wf"] = (
            u_normal_to_wall - u_normal_to_wall_true
        )
        self.equations["velocity_wall_parallel_wf"] = (
            u_parallel_to_wall_mag - u_parallel_to_wall_true
        )
        self.equations["k_wf"] = k - k_true
        self.equations["ep_wf"] = ep - ep_true
        self.equations["wall_shear_stress_x_wf"] = (
            wall_shear_stress_x - wall_shear_stress_true_x
        )
        self.equations["wall_shear_stress_y_wf"] = (
            wall_shear_stress_y - wall_shear_stress_true_y
        )
        # Adding the equation for wall shear stress in the z direction
        self.equations["wall_shear_stress_z_wf"] = (
            wall_shear_stress_z - wall_shear_stress_true_z
        )

class kEpsilonTransient(PDE):
    def __init__(self, nu=1, rho=1):
        # set params
        nu = Number(nu)
        rho = Number(rho)

        # coordinates
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")  # Add z coordinate
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}

        # velocity components
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        w = Function("w")(*input_variables)  # Add w velocity component
        p = Function("p")(*input_variables)
        k = Function("k")(*input_variables)
        ep = Function("ep")(*input_variables)

        # Model constants
        sig_k = Number(1.0)
        sig_ep = Number(1.3)
        C_ep1 = Number(1.44)
        C_ep2 = Number(1.92)
        C_mu = Number(0.09)
        E = Number(9.793)

        # Turbulent Viscosity
        nu_t = C_mu * (k ** 2) / (ep + 1e-4)

        # Turbulent Production Term
        P_k = nu_t * (
            2 * (u.diff(x)) ** 2
            + 2 * (v.diff(y)) ** 2
            + 2 * (w.diff(z)) ** 2  # Add w term
            + (u.diff(y) + v.diff(x)) ** 2
            + (u.diff(z) + w.diff(x)) ** 2  # Add w term
            + (v.diff(z) + w.diff(y)) ** 2  # Add w term
        )

        # set equations
        self.equations = {}
        self.equations["continuity"] = (u.diff(x) + v.diff(y) + w.diff(z))  # Modify for 3D
        self.equations["momentum_x"] = (
            rho * u.diff(t)
            + u * u.diff(x)
            + v * u.diff(y)
            + w * u.diff(z)  # Add w term
            + p.diff(x)
            - ((nu + nu_t) * u.diff(x)).diff(x)
            - ((nu + nu_t) * u.diff(y)).diff(y)
            - ((nu + nu_t) * u.diff(z)).diff(z)  # Add w term
        )
        """+ u * u.diff(x)
            + v * u.diff(y)
            + w * u.diff(z)  # Add w term
            + p.diff(x)
            - ((nu + nu_t) * u.diff(x)).diff(x)
            - ((nu + nu_t) * u.diff(y)).diff(y)
            - ((nu + nu_t) * u.diff(z)).diff(z)  # Add w term"""
        self.equations["momentum_y"] = (
            rho * v.diff(t)
            + u * v.diff(x)
            + v * v.diff(y)
            + w * v.diff(z)  # Add w term
            + p.diff(y)
            - ((nu + nu_t) * v.diff(x)).diff(x)
            - ((nu + nu_t) * v.diff(y)).diff(y)
            - ((nu + nu_t) * v.diff(z)).diff(z)  # Add w term
        )
        self.equations["momentum_z"] = (  # Add momentum equation for z
            rho * w.diff(t)
            + u * w.diff(x)
            + v * w.diff(y)
            + w * w.diff(z)
            + p.diff(z)
            - ((nu + nu_t) * w.diff(x)).diff(x)
            - ((nu + nu_t) * w.diff(y)).diff(y)
            - ((nu + nu_t) * w.diff(z)).diff(z)
        )
        self.equations["k_equation"] = (
            k.diff(t)
            + u * k.diff(x)
            + v * k.diff(y)
            + w * k.diff(z)  # Add w term
            - ((nu + nu_t / sig_k) * k.diff(x)).diff(x)
            - ((nu + nu_t / sig_k) * k.diff(y)).diff(y)
            - ((nu + nu_t / sig_k) * k.diff(z)).diff(z)  # Add z term
            - P_k
            + ep
        )
        self.equations["ep_equation"] = (
            ep.diff(t)
            + u * ep.diff(x)
            + v * ep.diff(y)
            + w * ep.diff(z)  # Add w term
            - ((nu + nu_t / sig_ep) * ep.diff(x)).diff(x)
            - ((nu + nu_t / sig_ep) * ep.diff(y)).diff(y)
            - ((nu + nu_t / sig_ep) * ep.diff(z)).diff(z)  # Add z term
            - (C_ep1 * P_k - C_ep2 * ep) * ep / (k + 1e-3)
        )