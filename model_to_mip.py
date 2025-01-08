import torch
from pprint import pprint
from torch import nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

import nfl_veripy.dynamics as dynamics  # noqa: E402
from nfl_veripy.utils.nn import load_controller  # noqa: E402
# controller = load_controller("GroundRobotSI", model_name="complex_potential_field")
controller = load_controller("Pendulum")
print(controller)

DEVICE = torch.device("cuda:0")
Dt = 0.1
# init_state = torch.tensor([[0.,0,0]]).view(-1,3).to(DEVICE)
init_state = torch.tensor([[1,0.01]]).to(DEVICE)
# init_state_range = torch.tensor([[-5.5,-4.5],[.5,1.5]])

eps = .1
norm = float("inf")
ptb = PerturbationLpNorm(norm = norm, eps = eps)
bounded_init_state = BoundedTensor(init_state, ptb)

lirpa_model = BoundedModule(controller, torch.empty_like(bounded_init_state), device=DEVICE)
lirpa_model.compute_bounds(x=(bounded_init_state,), method='alpha-CROWN')


save_dict = lirpa_model.save_intermediate('./controller_step1_bounds.pt')
print(save_dict)

lirpa_model.build_solver_module(model_type='mip')
opt = lirpa_model.solver_model
print(lirpa_model.solver_model)
# print(opt.getConstrs())
print(opt.write("./controller_step1.lp"))

from nfl_veripy.dynamics.Pendulum import PendulumDynamics
# dyn = dynamics.get_dynamics_instance("Pendulum", "FullState")
dyn = PendulumDynamics().to(DEVICE)

# import numpy as np

def step_1(xt):
    ut = controller(xt)
    # xt = xt.cpu()
    # xt1 = dyn.dynamics_step(xt, ut)
    xt1 = dyn(xt, ut)
    return xt1


# output = step_1(init_state)
# print(output)
# x = torch.cat([init_state, init_state], 0)
# print('x', x)
# output = step_1(x)
# print(output)
# exit()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_1step_samples_and_bounds(init_state_bounded, step_fn, bounds) -> None:
#   xt = np.random.uniform(low=initial_state_range[:, 0], high=initial_state_range[:, 1], size=(1000,2))
  xt = np.random.uniform(low=(init_state_bounded + init_state_bounded.ptb.eps).cpu(), high=(init_state_bounded - init_state_bounded.ptb.eps).cpu(), size=(1000,2))
  xt = torch.tensor(xt, dtype=torch.float).to(DEVICE)
  xt1 = step_fn(xt)
  # xt2 = step_fn(xt1)
  xt = xt.detach().cpu().numpy()
  xt1 = xt1.detach().cpu().numpy()
  # xt2 = xt2.detach().cpu().numpy()
  # print(xt)
  # print(xt1)
  plt.axes().set_aspect("equal")
  plt.plot(xt[:, 0], xt[:, 1], 'o', c="blue")
  plt.plot(xt1[:, 0], xt1[:, 1], 'o', c="red")
  # plt.plot(xt2[:, 0], xt2[:, 1], 'o', c="yellow")
  dims = [0, 1]
  if bounds is not None:
    rect = Rectangle(
        bounds[dims, 0],
        bounds[dims[0], 1] - bounds[dims[0], 0],
        bounds[dims[1], 1] - bounds[dims[1], 0],
        fc="None",
        linewidth=2,
        edgecolor="red",
    )
    plt.gca().add_patch(rect)
  plt.show()

plot_1step_samples_and_bounds(bounded_init_state, step_1, bounds=None)


u1_var = opt.getVarByName("lay/11_0")
u1_var.VarName = "u1"
x1_var = opt.getVarByName("inp_0")
x1_var.VarName = "x1"
x2_var = opt.getVarByName("inp_1")
x2_var.VarName = "x2"
opt.update()

import gurobipy as gp
from gurobipy import GRB

def find_BB(opt, var_name):
    opt.setObjective(opt.getVarByName(var_name), GRB.MINIMIZE)
    opt.optimize()
    lb = opt.ObjVal
    opt.setObjective(opt.getVarByName(var_name), GRB.MAXIMIZE)
    opt.optimize()
    ub = opt.ObjVal
    print(lb ,ub)
    return lb, ub

output_lb, output_ub = find_BB(opt, "u1")

from julia import OVERT, Main

Main.eval(f"""
using OVERT
pend_mass, pend_len, grav_const, friction = 0.5, 0.5, 1., 0.
func = :($(grav_const/pend_len) * sin(x1) + $(1/(pend_mass*pend_len^2)) * u1 - $(friction/(pend_mass*pend_len^2)) * x2)
range_dict = Dict(
    :u1 => [{output_lb}, {output_ub}],
    :x1 => [{init_state[0][0]-eps}, {init_state[0][0]+eps}],
    :x2 => [{init_state[0][1]-eps}, {init_state[0][1]+eps}]
)
oA = overapprox(func, range_dict, N=1, ϵ=0.0)
output_var_name = oA.output
""")

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

Main.eval("""
using JSON
constraints = []
for eq in oA.approx_eq
    json_str = JSON.json(eq)
    push!(constraints, json_str)
end
for ineq in oA.approx_ineq
    json_str = JSON.json(ineq)
    push!(constraints, json_str)
end
""")

constraints = Main.constraints

from AST_to_Gurobi import create_gurobi_model
model = create_gurobi_model(constraints, opt)
model.update()

print(model.write("model.lp"))

print(Main.output_var_name)

x2d_var = opt.getVarByName(Main.output_var_name)
x2d_var.VarName = "x2d"
x1d_var = opt.getVarByName("x2")

opt.update()

def get_or_create_var(opt_model, name):
    opt_model.update()
    var = opt_model.getVarByName(name)
    if var is None:
        var = opt_model.addVar(name=name, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    return var

x1_2_var = get_or_create_var(opt, "x1_2")
x2_2_var = get_or_create_var(opt, "x2_2")
opt.addConstr(x1_2_var == x1_var + Dt * x1d_var)
opt.addConstr(x2_2_var == x2_var + Dt * x2d_var)


x1_2_lb, x1_2_ub = find_BB(opt, "x1_2")
x2_2_lb, x2_2_ub = find_BB(opt, "x2_2")

print(x1_2_lb, x1_2_ub)
print(x2_2_lb, x2_2_ub)


plot_1step_samples_and_bounds(bounded_init_state, step_1, bounds=None)