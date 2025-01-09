import json
import gurobipy as gp
from gurobipy import GRB
from utils import get_or_create_var


def prase_ast_opt_model(ast, opt_model, aux_count):
    """
    Recursively parses a Julia AST (given as a Python dict) and converts it into a Gurobi expression.
    Note that this might result in adding some constraints to the model.

    :param ast: The dictionary representing the Julia AST.
    :param model: The Gurobi model to which constraints will be added.
    :return: A Gurobi variable or number corresponding to the AST. If its an expression such as max(), creates an aux var and returns it.
    """
    if isinstance(ast, (int, float)):
        return ast, aux_count
    elif isinstance(ast, str):
        return get_or_create_var(opt_model, ast), aux_count
    elif isinstance(ast, dict):
        head = ast.get('head')
        args = ast.get('args')

        assert head == "call"

        operator = args[0]
        inArgs = []
        for arg in args[1:]:
            var, aux_count = prase_ast_opt_model(arg, opt_model, aux_count) 
            inArgs.append(var)

        aux = opt_model.addVar(name=f"aux{aux_count}", lb=-GRB.INFINITY, ub=GRB.INFINITY)
        if operator == '+':
            opt_model.addConstr(aux == sum(inArgs), f"const{aux_count}")
        elif operator == '*':
            opt_model.addConstr(aux == inArgs[0] * inArgs[1], f"const{aux_count}")
        elif operator == '-':
            opt_model.addConstr(aux == inArgs[0] - inArgs[1], f"const{aux_count}") 
        elif operator == 'max':
            opt_model.addConstr(aux == gp.max_(inArgs[0], inArgs[1]), f"const{aux_count}")
        elif operator == 'min':
            opt_model.addConstr(aux == gp.min_(inArgs[0], inArgs[1]), f"const{aux_count}")
        else:
            raise ValueError(f"Unsupported operator: {operator}")
        # TODO: Change aux_name for const and eqs. 
        aux_count += 1
        return aux, aux_count
    else:
        raise ValueError(f"Unsupported AST node: {ast}")


    
def add_const_to_opt_model(eq, model, aux_count):
    head = eq.get('head')
    args = eq.get('args')

    assert head == "call"

    operator = args[0]
    inArgs = []
    for arg in args[1:]:
        var, aux_count = prase_ast_opt_model(arg, model, aux_count) 
        inArgs.append(var)

    assert len(inArgs) == 2

    if operator == '==':
        model.addConstr(inArgs[0] == inArgs[1], f"const{aux_count}")
    elif operator == '≦':
        model.addConstr(inArgs[0] <= inArgs[1], f"const{aux_count}")
    else:
        raise ValueError(f"Unsupported operator: {operator}")
    aux_count += 1
    return aux_count


def create_gurobi_model(constraints, opt_model):
    """
    Creates a Gurobi optimization model with the AST as a constraint.

    :param equations: A list with the equations in form of AST, the dictionary representing the Julia AST.
    :return: The created Gurobi model.
    """

    aux_count = 0
    for const in constraints:
        json_const = json.loads(const)
        aux_count = add_const_to_opt_model(json_const, opt_model, aux_count)

    # model.setObjective(vars["x"], GRB.MINIMIZE)

    return opt_model


if __name__ == "__main__":
    ast = {
        'args': ['+',
                {'args': ['*', -1.0, {'args': ['max', 0, {'args': ['*', -0.024096385542168676, {'args': ['-', 'x', -41.5], 'head': 'call'}], 'head': 'call'}], 'head': 'call'}], 'head': 'call'},
                {'args': ['*', -1.0, {'args': ['max', 0, {'args': ['min', {'args': ['*', 0.024096385542168676, {'args': ['-', 'x', -83.0], 'head': 'call'}], 'head': 'call'}, {'args': ['*', -0.024096385542168676, {'args': ['-', 'x', 0.0], 'head': 'call'}], 'head': 'call'}], 'head': 'call'}], 'head': 'call'}], 'head': 'call'}],
        'head': 'call'
    }

    model = create_gurobi_model(ast)
    model.update()

    print(model.printStats())
    print(model.display())
    print(model.write("model.lp"))

    # # Print the results
    # if model.status == GRB.OPTIMAL:
    #     print(f"Optimal solution found: x = {model.getVarByName('x').x}")
