from dataclasses import dataclass
from typing import List
import numpy as np
from numbers import Number
import matplotlib.pyplot as plt


@dataclass
class Operator:
    """
    A class that keeps the information for an operator, such as eventually (F), globally (G), and (&), or (|), negation (N) ...
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
    STEP = 0.1
    SIGNAL_START = 0
    SIGNAL_END = 10

    # The formula for the signal
    signal = lambda t: np.sin(t) + 0.5 * np.cos(2 * t)

    # spec = "G[0,10](x > 1.2)"
    # spec = "F[0,3](x < -1)"  # TODO: Start from 3
    spec = "G[0,5.5](F[0,4](x < 0.4))"

    # spec = "F[3,10](G[0,5](x > 7)) & G[0,8](y < 4 & z>2)"
    # spec = "((x < 1) | (y>2) ) & x<3"

    # Parses STL formula into python objects
    STL = STLParser(spec)
    print(STL.spec)


    # Visualization of the signal
    rng = STL.range(0, SIGNAL_END, STEP)
    plt.plot(rng, signal(rng))
    plt.show()

    # Printing the robustness degree
    print(STL.r(signal, STL.spec, SIGNAL_START))
