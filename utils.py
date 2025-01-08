import gurobipy as gp
from gurobipy import GRB, Model


def copy_constraints(model1, model2):
    """
    Copy variables from model2 to model1
    """
    # Add variables
    var_mapping = {}
    for var in model2.getVars():
        new_var = model1.addVar(
            lb=var.LB, ub=var.UB, obj=0, vtype=var.VType, name=f"{var.VarName}"
        )
        var_mapping[var] = new_var

    # Add simple constraints
    for constr in model2.getConstrs():
        expr = model2.getRow(constr)
        new_expr = sum(
            var_mapping[expr.getVar(i)] * expr.getCoeff(i) for i in range(expr.size())
        )
        model1.addConstr(
            new_expr, sense=constr.Sense, rhs=constr.RHS, name=f"{constr.ConstrName}"
        )

    # Add general constraints (min and max)
    for i in model2.getGenConstrs():
        if i.GenConstrType == GRB.GENCONSTR_MAX:
            constr = model2.getGenConstrMax(i)
            args = list(map(lambda x: var_mapping[x], constr[1]))
            if len(args) == 1:
                args.append(constr[2])
            model1.addConstr(var_mapping[constr[0]] == gp.max_(*args), i.GenConstrName)
        else:
            constr = model2.getGenConstrMin(i)
            args = list(map(lambda x: var_mapping[x], constr[1]))
            if len(args) == 1:
                args.append(constr[2])
            model1.addConstr(var_mapping[constr[0]] == gp.min_(*args), i.GenConstrName)

    return model1

if __name__ == "__main__":
    model2 = gp.read("model.lp")
    model1 = Model("merged_model")
    copy_constraints(model1, model2)
