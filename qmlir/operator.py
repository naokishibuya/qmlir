from typing import Tuple, Union
from .parameter import Parameter


# ----------------------------------------------------------------------------------------------------
# Operator Base Class
# ----------------------------------------------------------------------------------------------------


class Operator:
    def __init__(
        self,
        name: str,
        qubits: Tuple[int, ...],
        parameters: Tuple[Parameter, ...] = (),
        *,
        unitary: bool = True,  # U^† U = I
        hermitian: bool = False,  # U^† = U
        self_inverse: bool = False,  # U^2 = I
    ):
        # Add to the circuit
        from .circuit import QuantumCircuit

        circuit = QuantumCircuit.active_circuit()
        if circuit is not None:
            if not all(0 <= q < circuit.num_qubits for q in qubits):
                raise ValueError(f"Qubits {qubits} out of range for circuit with {circuit.num_qubits} qubits.")
            circuit.operators.append(self)

        self.name = name
        self.qubits = qubits
        self.parameters = parameters
        self.unitary = unitary
        self.hermitian = hermitian
        self.self_inverse = self_inverse

    def __repr__(self):
        qstr = ", ".join(f"{q}" for q in self.qubits)
        pstr = f"({', '.join(map(str, self.parameters))})" if self.parameters else ""
        return f"{self.name}{pstr}|{qstr}⟩"

    def __add__(self, other: Union["Operator", "OperatorComposition", "ObservableExpression"]):
        return ObservableExpression([(1.0, self)]) + other

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-1.0 * other)

    def __rsub__(self, other):
        return other + (-1.0 * self)

    def __matmul__(self, other: "Operator") -> "OperatorComposition":
        return OperatorComposition([self, other])

    def __rmul__(self, scalar: float) -> "ObservableExpression":
        return ObservableExpression([(scalar, self)])


# --------------------------------------------------------------------------------
# Composite Operator for Observables
# --------------------------------------------------------------------------------


class OperatorComposition:
    def __init__(self, terms):
        self.terms = terms

    def __repr__(self):
        return " @ ".join(map(str, self.terms))

    def __add__(self, other):
        return ObservableExpression([(1.0, self)]) + other

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-1.0 * other)

    def __rsub__(self, other):
        return (-1.0 * self) + other

    def __matmul__(self, other):
        return OperatorComposition(self.terms + [other])

    def __rmul__(self, scalar):
        return ObservableExpression([(scalar, self)])


class ObservableExpression:
    def __init__(self, terms):
        self.terms = terms  # List of (coefficient, Operator or OperatorComposition)

    def __repr__(self):
        return " + ".join(f"{coeff}*({term})" for coeff, term in self.terms)

    def __add__(self, other):
        if isinstance(other, ObservableExpression):
            return ObservableExpression(self.terms + other.terms)
        elif isinstance(other, (Operator, OperatorComposition)):
            return ObservableExpression(self.terms + [(1.0, other)])
        else:
            raise TypeError(f"Cannot add {other} to {self}")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ObservableExpression):
            negated = [(-c, t) for c, t in other.terms]
            return ObservableExpression(self.terms + negated)
        elif isinstance(other, (Operator, OperatorComposition)):
            return ObservableExpression(self.terms + [(-1.0, other)])
        else:
            raise TypeError(f"Cannot subtract {other} from {self}")

    def __rsub__(self, other):
        return (-1.0 * self) + other

    def __rmul__(self, scalar):
        return ObservableExpression([(scalar * coeff, term) for coeff, term in self.terms])


# --------------------------------------------------------------------------------
# Pauli operators
# --------------------------------------------------------------------------------


def I(*qubits):  # noqa: E743
    return Operator("I", qubits, hermitian=True, self_inverse=True)


def X(*qubits):
    return Operator("X", qubits, hermitian=True, self_inverse=True)


def Y(*qubits):
    return Operator("Y", qubits, hermitian=True, self_inverse=True)


def Z(*qubits):
    return Operator("Z", qubits, hermitian=True, self_inverse=True)


# --------------------------------------------------------------------------------
# Hadamard gate
# --------------------------------------------------------------------------------


def H(*qubits):
    return Operator("H", qubits, hermitian=True, self_inverse=True)


# --------------------------------------------------------------------------------
# Control gates
# --------------------------------------------------------------------------------


def CX(control, target):
    if control == target:
        raise ValueError("Control and target must differ.")
    return Operator("CX", (control, target), hermitian=True, self_inverse=True)


def CY(control, target):
    if control == target:
        raise ValueError("Control and target must differ.")
    return Operator("CY", (control, target), hermitian=True, self_inverse=True)


def CZ(control, target):
    if control == target:
        raise ValueError("Control and target must differ.")
    return Operator("CZ", (control, target), hermitian=True, self_inverse=True)


def CCX(control1, control2, target):
    if control1 == control2 or control1 == target or control2 == target:
        raise ValueError("All qubits must be distinct.")
    return Operator("CCX", (control1, control2, target), hermitian=True, self_inverse=True)


def CCY(control1, control2, target):
    if control1 == control2 or control1 == target or control2 == target:
        raise ValueError("All qubits must be distinct.")
    return Operator("CCY", (control1, control2, target), hermitian=True, self_inverse=True)


def CCZ(control1, control2, target):
    if control1 == control2 or control1 == target or control2 == target:
        raise ValueError("All qubits must be distinct.")
    return Operator("CCZ", (control1, control2, target), hermitian=True, self_inverse=True)


# --------------------------------------------------------------------------------
# Phase gates
# --------------------------------------------------------------------------------


def S(*qubits):
    return Operator("S", qubits)


def Sdg(*qubits):
    return Operator("Sdg", qubits)


def T(*qubits):
    return Operator("T", qubits)


def Tdg(*qubits):
    return Operator("Tdg", qubits)


# --------------------------------------------------------------------------------
# Rotation gates
# --------------------------------------------------------------------------------


def RX(theta):
    parameter = theta if isinstance(theta, Parameter) else Parameter(theta)
    return lambda *qubits: Operator("RX", qubits, (parameter,))


def RY(theta):
    parameter = theta if isinstance(theta, Parameter) else Parameter(theta)
    return lambda *qubits: Operator("RY", qubits, (parameter,))


def RZ(theta):
    parameter = theta if isinstance(theta, Parameter) else Parameter(theta)
    return lambda *qubits: Operator("RZ", qubits, (parameter,))
