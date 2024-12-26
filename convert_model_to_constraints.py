import torch
from pprint import pprint
from torch import nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


DEVICE = torch.device("cuda:0")
policy = torch.load("./ddpg_Pendulum-v1_ReLU.pth")
print(policy)

init_state = torch.tensor([[0.,0,0]]).view(-1,3).to(DEVICE)
eps = 1
norm = float("inf")
ptb = PerturbationLpNorm(norm = norm, eps = eps)
bounded_init_state = BoundedTensor(init_state, ptb)

lirpa_model = BoundedModule(policy, torch.empty_like(bounded_init_state), device=DEVICE)
lirpa_model.compute_bounds(x=(bounded_init_state,), method='alpha-CROWN')


save_dict = lirpa_model.save_intermediate('./ddpg_Pendulum-v1_intermediate_bounds.pt')
# print(save_dict)

lirpa_model.build_solver_module(model_type='mip')
opt = lirpa_model.solver_model
print(lirpa_model.solver_model)
# print(opt.getConstrs())
print(opt.write("./ddpg_Pendulum-v1.lp"))

# Named of last output is lay/12_0


# print(lirpa_model.named_modules)
# for i, node in lirpa_model.named_modules():
    # print(i)
    # print(node)
    # print(node.named_parameters())


output_lb, output_ub = save_dict["/12"]
# output_lb = torch.tensor([-83.])
# output_ub = torch.tensor([310.])






u1_var = opt.getVarByName("lay/12_0")
u1_var.VarName = "u1"
opt.update()





# -------------------------------------------------------------------------------------------------------------------------








# func = :(sin(x + y) * exp(z))
# range_dict = Dict(
#     :x => [1.0, 2.0],
#     :y => [-1.0, 1.0],
#     :z => [0.0, 1.0]
# )


from julia import OVERT, Main

Main.eval(f"""
using OVERT
func = :(tanh(u1))
range_dict = Dict(
    :u1 => [{output_lb.item()}, {output_ub.item()}],
)
result = overapprox(func, range_dict, N=1, ϵ=0.0)
""")

# overapprox = Main.result
# print(result)
# for i in range(len(overapprox.approx_eq)):
#     print(overapprox.approx_eq[i])
#     print(overapprox.approx_ineq[i])

Main.eval("""
using JSON
json_str = JSON.json(result.approx_eq[1].args[3])
""")

print(Main.json_str)

# print(overapprox.approx_eq[1])
# Main.eval(f"""
# dump(result.approx_eq[1].args[3])
#           """)
# print(Main.d)

# print(overapprox.approx_eq[1])
# print(overapprox.approx_eq[1].args[0])
# print(overapprox.approx_eq[1].args[1])
# print(overapprox.approx_eq[1].args[2])















# -------------------------------------------------------------------------------------------------------------------------







# json_str = '{"head":"call","args":["+",{"head":"call","args":["*",-1.0,{"head":"call","args":["max",0,{"head":"call","args":["*",-0.024096385542168676,{"head":"call","args":["-","x",-41.5]}]}]}]},{"head":"call","args":["*",-1.0,{"head":"call","args":["max",0,{"head":"call","args":["min",{"head":"call","args":["*",0.024096385542168676,{"head":"call","args":["-","x",-83.0]}]},{"head":"call","args":["*",-0.024096385542168676,{"head":"call","args":["-","x",0.0]}]}]}]}]},{"head":"call","args":["*",0.0,{"head":"call","args":["max",0,{"head":"call","args":["min",{"head":"call","args":["*",0.024096385542168676,{"head":"call","args":["-","x",-41.5]}]},{"head":"call","args":["*",-0.0064516129032258064,{"head":"call","args":["-","x",155.0]}]}]}]}]},{"head":"call","args":["*",1.0,{"head":"call","args":["max",0,{"head":"call","args":["min",{"head":"call","args":["*",0.0064516129032258064,{"head":"call","args":["-","x",0.0]}]},{"head":"call","args":["*",-0.0064516129032258064,{"head":"call","args":["-","x",310.0]}]}]}]}]},{"head":"call","args":["*",1.0,{"head":"call","args":["max",0,{"head":"call","args":["*",0.0064516129032258064,{"head":"call","args":["-","x",155.0]}]}]}]}]}'
json_str = Main.json_str

import json
json_data = json.loads(json_str)
pprint(json_data)

# def create_constraint(json_data):

from AST_to_Gurobi import create_gurobi_model
model = create_gurobi_model(json_data, opt, "u1")
model.update()

print(model.write("model.lp"))
