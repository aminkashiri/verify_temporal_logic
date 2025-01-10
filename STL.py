import numpy as np
from typing import List
import matplotlib.pyplot as plt
from dataclasses import dataclass
import gurobipy as gp


aux_name = lambda i: f"stl_aux{i}"
const_name = lambda i: f"stl_const{i}"


@dataclass
class Operator:
    """
    A class that keeps the information for an operator, such as eventually (F), globally (G), and (&), or (|), negation (N) ...
    It can also keep a predicate, for which name == "predicate".
    """

    name: str
    args: List
    # Only used for F and G
    start: int = None
    end: int = None

    def __str__(self) -> str:
        if self.name == "F" or self.name == "G":
            return f"{self.name}[{self.start},{self.end}]({self.args[0]})"
        elif self.name == "N":
            return f"N({self.args[0]})"
        elif self.name == "predicate":
            return f"{self.args[0]} {self.args[1]} {self.args[2]}"
        else:
            return "(" + f" {self.name} ".join(map(lambda x: str(x), self.args)) + ")"

    def create_aux(self, opt, start_step, info, used_predicates):
        self.opt = opt
        if self.name == "F":
            zs = []
            steps = int(self.end / info["dt"])
            for t in range(steps + 1):
                z = self.args[0].create_aux(opt, start_step + t, info, used_predicates)
                zs.append(z)

            return self.disjunct_zs(zs, info)
        elif self.name == "G":
            zs = []
            steps = int(self.end / info["dt"])
            for t in range(steps + 1):
                z = self.args[0].create_aux(opt, start_step + t, info, used_predicates)
                zs.append(z)

            return self.conjunct_zs(zs, info)
        elif self.name == "N":
            z = self.args[0].create_aux(opt, start_step, info, used_predicates)
            nz = self.opt.addVar(vtype=gp.GRB.BINARY, name=aux_name(info["aux_count"]))
            info["aux_count"] += 1
            self.opt.addConstr(nz == 1 - z, const_name(info["const_count"]))
            info["const_count"] += 1
            return nz
        elif self.name == "&":
            zs = [self.args[0], self.args[1]]
            return self.conjunct_zs(zs, info)
        elif self.name == "|":
            zs = [self.args[0], self.args[1]]
            return self.disjunct_zs(zs, info)
        elif self.name == "predicate":

            z = used_predicates.get(str(self) + f"_{start_step}")
            if z:
                return z

            #! Warn: get or create is only for test.
            from utils import get_or_create_var
            var = get_or_create_var(self.opt, self.args[0] + f"_{start_step}")

            # var = self.opt.getVarByName(self.args[0] + f"_{start_step}")

            # mu positive == satsify
            if self.args[1] == ">":
                mu = var - self.args[2]
            else:
                mu = -1 * (var - self.args[2])

            z = self.opt.addVar(vtype=gp.GRB.BINARY, name=aux_name(info["aux_count"]))
            info["aux_count"] += 1

            self.opt.addConstr(mu <= M * z - et, const_name(info["const_count"]))
            info["const_count"] += 1
            self.opt.addConstr(
                -1 * (mu) <= M * (1 - z) - et, const_name(info["const_count"])
            )
            info["const_count"] += 1

            used_predicates[str(self) + f"_{start_step}"] = z
            return z
        else:
            raise NotImplementedError("Operator not implemented")

    def conjunct_zs(self, zs, info):
        z_conj = self.opt.addVar(vtype=gp.GRB.BINARY, name=aux_name(info["aux_count"]))
        info["aux_count"] += 1
        sum_zs = 0
        for z in zs:
            self.opt.addConstr(z_conj <= z, const_name(info["const_count"]))
            info["const_count"] += 1
            sum_zs = sum_zs + z

        self.opt.addConstr(
            z_conj >= 1 - len(zs) + sum_zs, const_name(info["const_count"])
        )
        info["const_count"] += 1

        return z_conj

    def disjunct_zs(self, zs, info):
        z_disj = self.opt.addVar(vtype=gp.GRB.BINARY, name=aux_name(info["aux_count"]))
        info["aux_count"] += 1
        sum_zs = 0
        for z in zs:
            self.opt.addConstr(z_disj >= z, const_name(info["const_count"]))
            info["const_count"] += 1
            sum_zs = sum_zs + z

        self.opt.addConstr(z_disj <= sum_zs, const_name(info["const_count"]))
        info["const_count"] += 1

        return z_disj


class STLParser:
    def __init__(self, spec: str) -> None:
        spec = spec.replace(" ", "")
        self.spec = self.parse(spec)

    def find_ending_paranthesis_index(self, s: str):
        start_count = 0
        end_count = 0
        index = 0
        while index < len(s):
            if s[index] == "(":
                start_count += 1
            elif s[index] == ")":
                end_count += 1

            if end_count == start_count and start_count != 0:
                return index
            index += 1
        raise Exception("Error: Paranthesis never closed")

    def parse_F_or_G(self, s: str):
        start_t = float(s[2 : s.index(",")])
        end_t = float(s[s.index(",") + 1 : s.index("]")])
        arg_end_index = self.find_ending_paranthesis_index(s)
        arg = self.parse(s[s.index("]") + 2 : arg_end_index])
        operator = Operator(s[0], [arg], start_t, end_t)
        rest_of_spec = s[arg_end_index + 1 :]
        return operator, rest_of_spec

    def parse_predicate(self, spec: str):
        end = len(spec)
        if "&" in spec:
            end = spec.index("&")
        if "|" in spec and spec.index("|") < end:
            end = spec.index("|")

        predicate = spec[:end]

        operator = None
        operator_index = None
        if "<" in predicate:
            operator = "<"
            operator_index = predicate.index("<")
        elif ">" in predicate:
            operator = ">"
            operator_index = predicate.index(">")
        else:
            raise Exception("Error: Predicate has wrong form")

        var = predicate[:operator_index]
        value = float(predicate[operator_index + 1 :])
        operator = Operator("predicate", [var, operator, value])
        return operator, spec[end:]

    def parse_one_subspec(self, spec: str):
        if spec.startswith("F[") or spec.startswith("G["):
            operator, rest_of_spec = self.parse_F_or_G(spec)
        elif spec.startswith("N("):
            end_paranthesis_index = self.find_ending_paranthesis_index(spec)
            inside_operator = self.parse(spec[2:end_paranthesis_index])
            if inside_operator.name != "predicate":
                raise Exception("Error: Negation only before a predicate")
            operator = Operator("N", [inside_operator])
            rest_of_spec = spec[end_paranthesis_index + 1 :]
        elif spec.startswith("("):
            end_paranthesis_index = self.find_ending_paranthesis_index(spec)
            operator = self.parse(spec[1:end_paranthesis_index])
            rest_of_spec = spec[end_paranthesis_index + 1 :]
        else:
            operator, rest_of_spec = self.parse_predicate(spec)
        return operator, rest_of_spec

    def check_and_and_or(self, current_value, seen_operator):
        if not (seen_operator == "&" or seen_operator == "|"):
            raise Exception("Error: spec has wrong form")
        if current_value is None or current_value == seen_operator:
            return seen_operator
        else:
            raise Exception("Error: Spec is ambiguous")

    def parse(self, original_spec: str):
        """
        Parse a string of an specification into python objects
        """
        spec = original_spec
        current_operator = None
        args = []

        while True:
            first_operator, spec = self.parse_one_subspec(spec)
            args.append(first_operator)
            if len(spec) == 0:
                break
            current_operator = self.check_and_and_or(current_operator, spec[0])
            spec = spec[1:]

        if len(args) == 1:
            return args[0]
        else:
            return Operator(current_operator, args)

    def range(self, start, end, step):
        temp = np.arange(start, end + step, step)
        if temp[-1] > end:
            return temp[:-1]
        return temp

    def r(self, signal, spec, t):
        if spec.name == "predicate":
            if spec.args[1] == ">":
                return signal(t) - spec.args[2]
            else:
                return -1 * (signal(t) - spec.args[2])
        elif spec.name == "N":
            return -1 * self.r(signal, spec.args[0], t)
        elif spec.name == "&":
            return min(map(lambda x: self.r(signal, x, t), spec.args))
        elif spec.name == "|":
            return max(map(lambda x: self.r(signal, x, t), spec.args))
        elif spec.name == "G":
            return min(
                [
                    self.r(signal, spec.args[0], tp)
                    for tp in self.range(t + spec.start, t + spec.end, STEP)
                ]
            )
        elif spec.name == "F":
            return max(
                [
                    self.r(signal, spec.args[0], tp)
                    for tp in self.range(t + spec.start, t + spec.end, STEP)
                ]
            )
        raise Exception("Error: Should not be here")


if __name__ == "__main__":
    DT = 0.5

    # The formula for the signal
    # signal = lambda t: np.sin(t) + 0.5 * np.cos(2 * t)

    spec = "F[1,2](G[0,1](x1 > 3))"
    # spec = "G[0,10](x > 1.2)"
    # spec = "F[0,3](x < -1)"  # TODO: Start from 3
    # spec = "G[0,5.5](F[0,4](x < 0.4))"

    # spec = "F[3,10](G[0,5](x > 7)) & G[0,8](y < 4 & z>2)"
    # spec = "((x < 1) | (y>2) ) & x<3"

    # M = 1000000000
    # et = 0.000000001
    M = 1234
    et = 0.1

    # Parses STL formula into python objects
    spec = STLParser(spec).spec
    print(spec)

    opt = gp.Model()

    used_predicates = {}
    info = {"M": M, "et": et, "dt": DT, "aux_count": 0, "const_count": 0}
    v = spec.create_aux(opt, 1, info, used_predicates)
    opt.update()
    print(v)

    opt.write("test.lp")

    # Visualization of the signal
    # rng = STL.range(0, SIGNAL_END, STEP)
    # plt.plot(rng, signal(rng))
    # plt.show()

    # Printing the robustness degree
    # print(STL.r(signal, STL.spec, SIGNAL_START))
