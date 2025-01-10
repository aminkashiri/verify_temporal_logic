import torch
import gurobipy as gp
from julia import Main
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# import nfl_veripy.dynamics as dynamics
from nfl_veripy.utils.nn import load_controller

# from nfl_veripy.dynamics.Pendulum import PendulumDynamics
from dynamics import PendulumDynamics

from AST_to_Gurobi import create_gurobi_model
from utils import copy_constraints, get_or_create_var


def plot_bound(bound):
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


def plot_actual_samples(bounds, controller, dyn, steps, show_steps=False) -> None:
    plt.axes().set_aspect("equal")
    xt = torch.distributions.Uniform(bounds[0][:, 0], bounds[0][:, 1]).sample((1000,))
    for step in range(steps):
        ut = controller(xt)
        xt1 = dyn(xt, ut)
        if step == 0:
            xt_np = xt.detach().cpu().numpy()
            plt.plot(xt_np[:, 0], xt_np[:, 1], "o", c="blue")

        xt1_np = xt1.detach().cpu().numpy()
        if step < steps - 1 and show_steps:
            plt.plot(xt1_np[:, 0], xt1_np[:, 1], "o", c="black")
        elif step == steps - 1:
            plt.plot(xt1_np[:, 0], xt1_np[:, 1], "o", c="black")

        xt = xt1

    # for i, bound in enumerate(bounds):
    #     if i == 0 or i == len(bounds) - 1 or show_steps:
    #         plot_bound(bound)
    plt.show()


def plot_recursive_samples_and_bounds(
    bounds, controller, dyn, show_steps=False
) -> None:
    plt.axes().set_aspect("equal")
    for i, bound in enumerate(bounds):
        xt = torch.distributions.Uniform(bound[:, 0], bound[:, 1]).sample((1000,))
        ut = controller(xt)
        xt1 = dyn(xt, ut)
        xt = xt.detach().cpu().numpy()
        xt1 = xt1.detach().cpu().numpy()
        if i == 0:
            plt.plot(xt[:, 0], xt[:, 1], "o", c="blue")

        if i < len(bounds) - 2:
            if show_steps:
                plt.plot(xt1[:, 0], xt1[:, 1], "o", c="red")
        elif i == len(bounds) - 2:
            plt.plot(xt1[:, 0], xt1[:, 1], "o", c="red")

        if i == 0 or i == len(bounds) - 1 or show_steps:
            plot_bound(bound)
    plt.show()


def find_BB(opt, var_name):
    # TODO: Can I use Lirpa dict?
    opt.setObjective(opt.getVarByName(var_name), gp.GRB.MINIMIZE)
    opt.optimize()
    lb = opt.ObjVal
    opt.setObjective(opt.getVarByName(var_name), gp.GRB.MAXIMIZE)
    opt.optimize()
    ub = opt.ObjVal
    return lb, ub


def add_one_step_constraints(
    controller, dyn, init_state_range, step, device, dt, show=False
):
    init_state = (init_state_range[:, 1] + init_state_range[:, 0]).unsqueeze(0) / 2.0
    eps = (init_state_range[:, 1] - init_state_range[:, 0]) / 2.0
    ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
    bounded_init_state = BoundedTensor(init_state, ptb)

    lirpa_model = BoundedModule(controller, torch.empty_like(init_state), device=device)
    lirpa_model.compute_bounds(x=(bounded_init_state,), method="alpha-CROWN")

    # save_dict = lirpa_model.save_intermediate("./controller_step1_bounds.pt")
    # print(save_dict)

    lirpa_model.build_solver_module(model_type="mip")
    step_opt = lirpa_model.solver_model

    u1_var = step_opt.getVarByName("lay/11_0")
    u1_var.VarName = f"u1_{step}"
    x1_1_var = step_opt.getVarByName("inp_0")
    x1_1_var.VarName = f"x1_{step}"
    x2_1_var = step_opt.getVarByName("inp_1")
    x2_1_var.VarName = f"x2_{step}"
    step_opt.update()

    control_lb, control_ub = find_BB(step_opt, f"u1_{step}")

    Main.eval(
        f"""
    using OVERT
    using JSON

    pend_mass, pend_len, grav_const, friction = {dyn.mass}, {dyn.length}, {dyn.gravity}, 0.
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

    step_opt = create_gurobi_model(Main.constraints, step_opt)
    step_opt.update()

    x1d_var = step_opt.getVarByName(f"x2_{step}")
    x2d_var = step_opt.getVarByName(Main.output_var_name)
    x2d_var.VarName = f"x2d_{step}"

    step_opt.update()

    x1_2_var = get_or_create_var(step_opt, f"x1_{step+1}")
    x2_2_var = get_or_create_var(step_opt, f"x2_{step+1}")
    step_opt.addConstr(x1_2_var == x1_1_var + dt * x1d_var)
    step_opt.addConstr(x2_2_var == x2_1_var + dt * x2d_var)

    step_opt.write(f"step_{step}_opt.lp")

    x1_2_lb, x1_2_ub = find_BB(step_opt, f"x1_{step+1}")
    x2_2_lb, x2_2_ub = find_BB(step_opt, f"x2_{step+1}")

    # print(x1_2_lb, x1_2_ub)
    # print(x2_2_lb, x2_2_ub)
    next_state_range = torch.tensor(
        [[x1_2_lb, x1_2_ub], [x2_2_lb, x2_2_ub]], device=device
    )

    if show:
        plot_recursive_samples_and_bounds(
            [init_state_range, next_state_range], controller, dyn
        )
    return step_opt, next_state_range


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0")
    DT = 0.1
    EPS = 0.1
    STEPS = 4

    mass = 0.5
    length = 0.5
    gravity = 1
    friction = 0

    controller = load_controller("Pendulum").to(DEVICE)
    dyn = PendulumDynamics(DT, gravity, length, mass).to(DEVICE)
    # dyn = dynamics.get_dynamics_instance("Pendulum", "FullState")
    init_state = torch.tensor([[1, 0.01]]).to(DEVICE)
    init_state_range = torch.tensor(
        [
            [init_state[0][0] - EPS, init_state[0][0] + EPS],
            [init_state[0][1] - EPS, init_state[0][1] + EPS],
        ]
    ).to(DEVICE)

    state_bounds = [init_state_range]
    main_opt = gp.Model("merged_model")
    for step in range(STEPS):
        step_opt, next_state_range = add_one_step_constraints(
            controller,
            dyn,
            state_bounds[-1],
            step + 1,
            DEVICE,
            DT,
            # True if step + 1 == STEPS else False,
            False,
        )
        copy_constraints(main_opt, step_opt, postfix=f"_{step+1}")
        state_bounds.append(next_state_range)

    plot_actual_samples(state_bounds, controller, dyn, STEPS)
    plot_recursive_samples_and_bounds(state_bounds, controller, dyn)
    main_opt.write("main_opt.lp")

    x1_t_lb, x1_t_ub = find_BB(main_opt, f"x1_{STEPS+1}")
    x2_t_lb, x2_t_ub = find_BB(main_opt, f"x2_{STEPS+1}")
    print("-------------------------------------------------------------------")
    print("Recursive result", state_bounds[-1])
    print("One shot results:\n ", x1_t_lb, x1_t_ub, "\n", x2_t_lb, x2_t_ub)
