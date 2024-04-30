import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

from sympy import Symbol, Eq, Abs, sin, cos, And, Or, Number, Function, simplify, exp, Min, Max, log, sqrt
from modulus.eq.pde import PDE

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.utils.io import csv_to_dict
from modulus.solver import SequentialSolver
from modulus.domain import Domain
from modulus.geometry.tessellation import Tessellation


from modulus.loss.loss import CausalLossNorm


from modulus.geometry.primitives_3d import Box
from modulus.geometry.parameterization import OrderedParameterization

from modulus.models.fully_connected import FullyConnectedArch
from modulus.models.moving_time_window import MovingTimeWindowArch
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.domain.inferencer import PointVTKInferencer
from modulus.utils.io import (
    VTKUniformGrid,
)
from modulus.key import Key
from modulus.node import Node
from modulus.eq.pdes.navier_stokes import NavierStokes

from custom_k_ep_3D import kEpsilonTransient, kEpsilonInit, kEpsilonLSWF

#set working dir to current - doesn't work in docker
"""
print(os.getcwd())
abspath = os.path.abspath(__file__)
print(abspath)
dname = os.path.dirname(abspath)
print(dname)
os.chdir(dname)
print(os.getcwd())
"""

@modulus.main(config_path="conf", config_name="config") #config_fourier
def run(cfg: ModulusConfig) -> None:

    # time window parameters
    nu = 0.000015
    resolved_y_start = 30 * nu  # what is this???
    time_window_size = 1.0
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, time_window_size)}
    nr_time_windows = 10

    # make navier stokes equations - air of 20 deg C - nu=0.000015, rho=1.2
    # ns = NavierStokes(nu=0.000015, rho=1.2, dim=3, time=True)
    init = kEpsilonInit(nu=nu, rho=1.2)
    ns = kEpsilonTransient(nu=nu, rho=1.2)
    wf = kEpsilonLSWF(nu=nu, rho=1.2, y_plus=resolved_y_start)

    # define sympy variables to parametrize domain curves
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

    # make geometry for problem

    print(os.getcwd())
    # for now, just consider the blades
    geom_path = "/root/Desktop/workspace/examples/taylor_green/geometry" # absolute due to troubles with docker
    blades = Tessellation.from_stl(geom_path + "/blades.stl", airtight=True,
        parameterization=OrderedParameterization(time_range, key=t_symbol))

    # normalize meshes
    def normalize_mesh(mesh, center, scale):
        mesh = mesh.translate([-c for c in center])
        mesh = mesh.scale(scale) # not important right now
        return mesh

    idx = 0
    bnds = {}
    for key, value in blades.bounds.bound_ranges.items():
        bnds[blades.dims[idx]] = value
        idx += 1

    #  normalization
    def get_center(a, b):
        return ( a + b ) / 2

    def scale_dict_values(input_dict, scale_factor):
        for key in input_dict:
            if isinstance(input_dict[key], (int, float)):
                input_dict[key] *= scale_factor
        return input_dict

    x_offset = get_center(bnds['x'][0],bnds['x'][1])
    y_offset = get_center(bnds['y'][0],bnds['y'][1])
    z_offset = get_center(bnds['z'][0],bnds['z'][1])
    center = (x_offset, y_offset, z_offset)
    scale = 0.1
    blades = normalize_mesh(blades, center, scale)

    bnds = {}
    idx = 0
    for key, value in blades.bounds.bound_ranges.items():
        bnds[blades.dims[idx]] = value
        idx += 1

    #bnds = scale_dict_values(bnds, scale)
    thickness = [(bnds[x][1]-bnds[x][0])/2 for x in ['x', 'y', 'z']]
    print(thickness)
    # expecting that the geometry center is now set to (0, 0, 0)
    # doing some random magic here, according to the geometry size
    # must be rounded so we do not get "geometry has no surface" error
    # there is problem with sampling if size is float
    # also we need to create a tunnel with reasonable geometry, so if one side is by far different, adjust it...
    if (thickness[0]/thickness[1]>6):
        print("adjusting thickness")
        thickness[1] = thickness[0]
    if (thickness[1]/thickness[0]>6):
        print("adjusting thickness")
        thickness[0] = thickness[1]
    channel_length = (round(bnds['y'][0]-thickness[1]), round(bnds['y'][1]+thickness[1]*2))
    channel_width = (round(bnds['x'][0]-thickness[0]), round(bnds['x'][1]+thickness[0]))
    channel_height = (round(bnds['z'][0]-thickness[2]), round(bnds['z'][1]+thickness[2]))
    box_bounds = {x: channel_width, y: channel_length, z: channel_height}
    print(box_bounds)

    # calculating the center of inlet:
    center_x = (bnds['x'][0] + bnds['x'][1]) / 2
    center_z = (bnds['z'][0] + bnds['z'][1]) / 2
    center_of_inlet = (center_x, bnds['y'][0], center_z)

    # define interior geometry, without blades
    rec = Box(
        (channel_width[0], channel_length[0], channel_height[0]),
        (channel_width[1], channel_length[1], channel_height[1]),
        parameterization=OrderedParameterization(time_range, key=t_symbol)
    ) #+ blades

    geo = rec + blades
    # uncomment for debugging
    """
    sample_dictionary = geo.sample_boundary(nr_points=50, criteria=And((y > channel_length[0]), 
                                                                        (y < channel_length[1]), 
                                                                        Or(
                                                                            Or(Eq(x, channel_width[0]), Eq(x, channel_width[1])), 
                                                                            Or(Eq(z, channel_height[0]), Eq(z, channel_height[1]))
                                                                            )
                                                                        ))
    print(sample_dictionary)
    """
    print(rec.bounds)
    print(geo.bounds)
    print(blades.bounds)

    # this function does not work properly right now, needs fix
    def rectangular_inlet(x, y, z, center, max_vel):
        """
        This function defines the velocity profile for a square inlet.

        Args:
            x: Float, x-coordinate of the point.
            y: Float, y-coordinate of the point.
            z: Float, z-coordinate of the point (unused in this case).
            center: List of floats, center coordinates of the square inlet.
            max_vel: Float, maximum velocity at the inlet.

        Returns:
            List of floats: The velocity components (u, v, w) at the point.
        """
        centered_x = x - center[0]
        centered_z = z - center[2]
        print("center:", center)

        width = channel_width[1] - channel_width[0]
        height = channel_height[1] - channel_height[0]

        distance_from_wall = Max(Abs(centered_x / (width/2.2)) , Abs(centered_z / (height/2.2))) #the higher the distance in comparison to threshold, the worse; 2.2 as we want it to be 10% near wall

        velocity = max_vel / Max(distance_from_wall, 1)  # experimental function

        return 0 * velocity, velocity, 0 * velocity

    u, v, w = rectangular_inlet(x, y, z, center_of_inlet, 10)

    # make network for current step and previous step
    #"""
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("w")],
        layer_size=256,
    )

    k_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("k_star")],
        layer_size=256,
    )
    ep_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("ep_star")],
        layer_size=256,
    )
    p_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("p")],
        layer_size=256,
    )

    """
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p"), Key("k_star"), Key("ep_star")],
        frequencies=("axis", [i / 2 for i in range(8)]),
        frequencies_params=("axis", [i / 2 for i in range(8)]),
        cfg=cfg.arch.fourier,
    )
    """

    time_window_net = MovingTimeWindowArch(flow_net, time_window_size)
    time_window_net_k = MovingTimeWindowArch(k_net, time_window_size)
    time_window_net_p = MovingTimeWindowArch(p_net, time_window_size)
    time_window_net_ep = MovingTimeWindowArch(ep_net, time_window_size)

    # make nodes to unroll graph on
    nodes = (init.make_nodes() 
            + ns.make_nodes() 
            + wf.make_nodes()
            + [time_window_net.make_node(name="time_window_network")]
            + [time_window_net_k.make_node(name="time_window_network_k")]
            + [time_window_net_p.make_node(name="time_window_network_p")]
            + [time_window_net_ep.make_node(name="time_window_network_ep")]
            + [Node.from_sympy(Min(log(1 + exp(Symbol("k_star"))) + 1e-4, 20), "k")]
            + [Node.from_sympy(Min(log(1 + exp(Symbol("ep_star"))) + 1e-4, 180), "ep")])

    # make initial condition domain
    ic_domain = Domain("initial_conditions")

    # make moving window domain
    window_domain = Domain("window")

    # add constraints to solver
    p_grad = 1.0
    # inlet BC
    # works well
    inletBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        # outvar={"p": p_grad * (channel_length[1] - channel_length[0])},
        # lambda_weighting={"p": 10},
        outvar={"u": u, "v": v, "w": w},
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={"u": 100, "v": 1000, "w": 100},
        criteria=Eq(y, channel_length[0]),
        parameterization=time_range,
    )
    ic_domain.add_constraint(inletBC, "inletBC")
    window_domain.add_constraint(inletBC, "inletBC")

    # outlet BC
    # works well
    outletBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"p" : 0},
        batch_size=cfg.batch_size.initial_condition,
        criteria=Eq(y, channel_length[1]),
        parameterization=time_range,
    )
    ic_domain.add_constraint(outletBC, "outletBC")
    window_domain.add_constraint(outletBC, "outletBC")

    # tunnel walls
    # works well
    """
    noslipBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={"u": 100, "v": 100, "w": 100},
        parameterization=time_range,
        # criteria for all side walls
        criteria=And((y > channel_length[0]), 
                    (y < channel_length[1]), 
                    Or(
                        Or(Eq(x, channel_width[0]), Eq(x, channel_width[1])), 
                        Or(Eq(z, channel_height[0]), Eq(z, channel_height[1]))
                    )
                ),
    )
    ic_domain.add_constraint(noslipBC, "noslipBC")
    window_domain.add_constraint(noslipBC, "noslipBC")
    """

    # Point where wall funciton is applied
    wf_pt = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={
            # "u": 0, "v": 0, "w": 0,
            "velocity_wall_normal_wf": 0,
            "velocity_wall_parallel_wf": 0,
            "ep_wf": 0,
            "wall_shear_stress_x_wf": 0,
            "wall_shear_stress_y_wf": 0,
            "wall_shear_stress_z_wf": 0,
        },
        lambda_weighting={
            # "u": 100, "v": 100, "w": 100,
            "velocity_wall_normal_wf": 100,
            "velocity_wall_parallel_wf": 100,
            "ep_wf": 1,
            "wall_shear_stress_x_wf": 100,
            "wall_shear_stress_y_wf": 100,
            "wall_shear_stress_z_wf": 100,
        },
        batch_size=cfg.batch_size.initial_condition,
        # criteria for all side walls
        criteria=And((y > channel_length[0]), 
                    (y < channel_length[1]), 
                    Or(
                        Or(Eq(x, channel_width[0]), Eq(x, channel_width[1])), 
                        Or(Eq(z, channel_height[0]), Eq(z, channel_height[1]))
                    )
                ),
        parameterization=time_range,
    )
    ic_domain.add_constraint(wf_pt, "WF")
    window_domain.add_constraint(wf_pt, "WF")

    
    # blade geometry BC
    """
    bladesBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=blades,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={"u": 100, "v": 100, "w": 100},
        parameterization=time_range,
    )
    ic_domain.add_constraint(bladesBC, "bladesBC")
    window_domain.add_constraint(bladesBC, "bladesBC")
    """
    wf_pt_blades = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=blades,
        outvar={
            "u": 0, "v": 0, "w": 0,
            "velocity_wall_normal_wf": 0,
            "velocity_wall_parallel_wf": 0,
            "ep_wf": 0,
            "wall_shear_stress_x_wf": 0,
            "wall_shear_stress_y_wf": 0,
            "wall_shear_stress_z_wf": 0,
        },
        lambda_weighting={
            "u": 100, "v": 100, "w": 100,
            "velocity_wall_normal_wf": 100,
            "velocity_wall_parallel_wf": 100,
            "ep_wf": 1,
            "wall_shear_stress_x_wf": 100,
            "wall_shear_stress_y_wf": 100,
            "wall_shear_stress_z_wf": 100,
        },
        batch_size=cfg.batch_size.initial_condition,
        parameterization=time_range,
    )
    ic_domain.add_constraint(wf_pt_blades, "WF_blades")
    window_domain.add_constraint(wf_pt_blades, "WF_blades")

    ic = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        bounds=box_bounds,
        outvar={
            "continuity": 0,
            "momentum_x": 0,
            "momentum_y": 0,
            "momentum_z": 0,
            "k_equation": 0,
            "ep_equation": 0,
        },
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={
            "continuity": 100,
            "momentum_x": 1000,
            "momentum_y": 1000,
            "momentum_z": 1000,
            "k_equation": 10,
            "ep_equation": 1,
        },
        parameterization=time_range,
    )
    ic_domain.add_constraint(ic, name="ic")
    window_domain.add_constraint(ic, "ic")

    # flow initialization
    flow = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        bounds=box_bounds,
        outvar={"u_init": 0, "v_init": 0, "k_init": 0, "p_init": 0, "ep_init": 0},
        batch_size=cfg.batch_size.interior_init,
        parameterization={t_symbol: 0},
    )
    ic_domain.add_constraint(flow, "FlowInit")
    # window_domain.add_constraint(interior, name="FlowInit") # I think this has to be always set

    # interior initialization
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        bounds=box_bounds,
        outvar={"u": 0, "v": 0, "w": 0, "p": 0},
        batch_size=cfg.batch_size.interior_init,
        lambda_weighting={"u": 100, "v": 1000, "w": 100, "p": 100},
        parameterization={t_symbol: 0},
    )
    ic_domain.add_constraint(interior, "InteriorInit")

    # make constraint for matching previous windows initial condition
    
    ic_diff = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={
            "u_prev_step_diff": 0, 
            "v_prev_step_diff": 0, 
            "w_prev_step_diff": 0,
        },
        batch_size=cfg.batch_size.interior,
        bounds=box_bounds,
        lambda_weighting={
            "u_prev_step_diff": 100,
            "v_prev_step_diff": 100,
            "w_prev_step_diff": 100,
        },
        parameterization={t_symbol: 0},
    )
    window_domain.add_constraint(ic_diff, name="ic_diff")

    # add inference data for time slices
    for i, specific_time in enumerate(np.linspace(0, time_window_size, 10)):
        vtk_obj = VTKUniformGrid(
            bounds=[channel_width, channel_length, channel_height],
            npoints=[64, 64, 64],
            export_map={"u": ["u", "v", "w"], "p": ["p"]},
        )
        grid_inference = PointVTKInferencer(
            vtk_obj=vtk_obj,
            nodes=nodes,
            input_vtk_map={"x": "x", "y": "y", "z": "z"},
            output_names=["u", "v", "w", "p"],
            requires_grad=False,
            invar={"t": np.full([64 ** 3, 1], specific_time)},
            batch_size=10000,
        )
        ic_domain.add_inferencer(grid_inference, name="time_slice_" + str(i).zfill(4))
        window_domain.add_inferencer(
            grid_inference, name="time_slice_" + str(i).zfill(4)
        )

    # make solver
    slv = SequentialSolver(
        cfg,
        [(1, ic_domain), (nr_time_windows, window_domain)],
        custom_update_operation=time_window_net.move_window,
    )

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()