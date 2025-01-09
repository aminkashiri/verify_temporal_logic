import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB, Model
from julia import OVERT, Main
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import nfl_veripy.dynamics as dynamics
from nfl_veripy.utils.nn import load_controller
from nfl_veripy.dynamics.Pendulum import PendulumDynamics

from AST_to_Gurobi import create_gurobi_model


def plot_1step_samples_and_bounds(
    bounded_init_state, controller, dyn, bound=None
) -> None:
    xt = np.random.uniform(
        low=(bounded_init_state + bounded_init_state.ptb.eps).cpu(),
        high=(bounded_init_state - bounded_init_state.ptb.eps).cpu(),
        size=(1000, 2),
    )
    xt = torch.tensor(xt, dtype=torch.float).to(DEVICE)
    ut = controller(xt)
    xt1 = dyn(xt, ut)
    xt = xt.detach().cpu().numpy()
    xt1 = xt1.detach().cpu().numpy()
    plt.axes().set_aspect("equal")
    plt.plot(xt[:, 0], xt[:, 1], "o", c="blue")
    plt.plot(xt1[:, 0], xt1[:, 1], "o", c="red")

    if bound is not None:
        lb_ub = bound.cpu().numpy()
        rect = Rectangle(
            lb_ub[:, 0],
            lb_ub[0, 1] - lb_ub[0, 0],
            lb_ub[1, 1] - lb_ub[1, 0],
            fc="None",
            linewidth=2,
            edgecolor="red",
        )
        plt.gca().add_patch(rect)
    plt.show()


def get_or_create_var(opt_model, name):
    opt_model.update()
    var = opt_model.getVarByName(name)
    if var is None:
        var = opt_model.addVar(name=name, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    return var


def find_BB(opt, var_name):
    # TODO: Can I use Lirpa dict?
    opt.setObjective(opt.getVarByName(var_name), GRB.MINIMIZE)
    opt.optimize()
    lb = opt.ObjVal
    opt.setObjective(opt.getVarByName(var_name), GRB.MAXIMIZE)
    opt.optimize()
    ub = opt.ObjVal
    print(lb, ub)
    return lb, ub


def add_one_step_constraints(controller, dyn, init_state_range, step):
    init_state = (init_state_range[:, 1] + init_state_range[:, 0]).unsqueeze(0) / 2.0
    eps = (init_state_range[:, 1] - init_state_range[:, 0]) / 2.0
    ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
    bounded_init_state = BoundedTensor(init_state, ptb)

    lirpa_model = BoundedModule(controller, torch.empty_like(init_state), device=DEVICE)
    lirpa_model.compute_bounds(x=(bounded_init_state,), method="alpha-CROWN")

    # save_dict = lirpa_model.save_intermediate("./controller_step1_bounds.pt")
    # print(save_dict)

    lirpa_model.build_solver_module(model_type="mip")
    step_opt = lirpa_model.solver_model

    plot_1step_samples_and_bounds(
        bounded_init_state, controller, dyn, bound=init_state_range
    )

    u1_var = step_opt.getVarByName("lay/11_0")
    u1_var.VarName = f"u1_{step}"
    x1_var = step_opt.getVarByName("inp_0")
    x1_var.VarName = f"x1_{step}"
    x2_var = step_opt.getVarByName("inp_1")
    x2_var.VarName = f"x2_{step}"
    step_opt.update()

    control_lb, control_ub = find_BB(step_opt, f"u1_{step}")

    Main.eval(
        f"""
    using OVERT
    using JSON

    pend_mass, pend_len, grav_const, friction = 0.5, 0.5, 1., 0.
    func = :($(grav_const/pend_len) * sin(x1_{step}) + $(1/(pend_mass*pend_len^2)) * u1_{step} - $(friction/(pend_mass*pend_len^2)) * x2_{step})
    range_dict = Dict(
        :u1_{step} => [{control_lb}, {control_ub}],
        :x1_{step} => [{init_state_range[0][0]}, {init_state_range[0][1]}],
        :x2_{step} => [{init_state_range[1][0]}, {init_state_range[1][1]}],
    )
    oA = overapprox(func, range_dict, N=1, ϵ=0.0)
    output_var_name = oA.output
    constraints = []
    for eq in oA.approx_eq
        json_str = JSON.json(eq)
        push!(constraints, json_str)
    end
    for ineq in oA.approx_ineq
        json_str = JSON.json(ineq)
        push!(constraints, json_str)
    end
    """
    )

    # overapprox = Main.result
    # print(overapprox)
    # print("---------------")
    # for i in range(len(overapprox.approx_eq)):
    #     print("--------------- eq ", i)
    #     print(overapprox.approx_eq[i])
    # for i in range(len(overapprox.approx_ineq)):
    #     print("--------------- ineq ", i)
    #     print(overapprox.approx_ineq[i])
    # print("---------------")

    step_opt = create_gurobi_model(Main.constraints, step_opt)
    step_opt.update()

    step_opt.write(f"step_{step}_model.lp")

    x1d_var = step_opt.getVarByName(f"x2_{step}")
    x2d_var = step_opt.getVarByName(Main.output_var_name)
    x2d_var.VarName = f"x2d_{step}"

    step_opt.update()

    x1_2_var = get_or_create_var(step_opt, f"x1_{step+1}")
    x2_2_var = get_or_create_var(step_opt, f"x2_{step+1}")
    step_opt.addConstr(x1_2_var == x1_var + DT * x1d_var)
    step_opt.addConstr(x2_2_var == x2_var + DT * x2d_var)

    x1_2_lb, x1_2_ub = find_BB(step_opt, f"x1_{step+1}")
    x2_2_lb, x2_2_ub = find_BB(step_opt, f"x2_{step+1}")

    print(x1_2_lb, x1_2_ub)
    print(x2_2_lb, x2_2_ub)
    next_state_range = torch.tensor(
        [[x1_2_lb, x1_2_ub], [x2_2_lb, x2_2_ub]], device=DEVICE
    )

    plot_1step_samples_and_bounds(
        bounded_init_state, controller, dyn, bound=next_state_range
    )
    return next_state_range


if __name__ == "__main__":
    # controller = load_controller("GroundRobotSI", model_name="complex_potential_field")
    controller = load_controller("Pendulum")
    print(controller)

    DEVICE = torch.device("cuda:0")
    DT = 0.1
    EPS = 0.1

    # dyn = dynamics.get_dynamics_instance("Pendulum", "FullState")
    dyn = PendulumDynamics().to(DEVICE)
    init_state = torch.tensor([[1, 0.01]]).to(DEVICE)
    init_state_range = torch.tensor(
        [
            [init_state[0][0] - EPS, init_state[0][0] + EPS],
            [init_state[0][1] - EPS, init_state[0][1] + EPS],
        ]
    ).to(DEVICE)

    next_state_range = add_one_step_constraints(controller, dyn, init_state_range, 1)
    next_state_range = add_one_step_constraints(controller, dyn, next_state_range, 2)
