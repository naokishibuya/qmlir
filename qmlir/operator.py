from typing import Tuple
from .bit import QuantumBit
from .parameter import Parameter


# ----------------------------------------------------------------------------------------------------
# Operator Base Class
# ----------------------------------------------------------------------------------------------------


class Operator:
    def __init__(
        self,
        name: str,
        qubits: Tuple[QuantumBit, ...],
        parameters: Tuple[Parameter, ...] = (),
        *,
        unitary: bool = True,  # U^† U = I
        hermitian: bool = False,  # U^† = U
        inverse_operator=None,  # Function to compute the inverse of this operator
    ):
        # Validate parameters
        if not all(isinstance(p, Parameter) for p in parameters):
            raise TypeError("Operator parameters must be instances of Parameter.")

        self.name = name
        self.qubits = qubits
        self.parameters = parameters
        self.unitary = unitary
        self.hermitian = hermitian
        self.inverse_operator = inverse_operator

        # add to the circuit
        circuit = qubits[0].circuit
        if not all(q.circuit == circuit for q in qubits):
            raise ValueError("All qubits must belong to the same quantum circuit.")
        circuit.operators.append(self)

    def inverse(self):
        return self.inverse_operator(*self.qubits) if self.inverse_operator else self

    def __repr__(self):
        qstr = ", ".join(f"q{q.index}" for q in self.qubits)
        pstr = f"({', '.join(map(str, self.parameters))})" if self.parameters else ""
        return f"{self.name}{pstr}|{qstr}⟩"


# --------------------------------------------------------------------------------
# Pauli operators
# --------------------------------------------------------------------------------


def I(*qubits):  # noqa: E743
    return Operator("I", qubits, hermitian=True)


def X(*qubits):
    return Operator("X", qubits, hermitian=True)


def Y(*qubits):
    return Operator("Y", qubits, hermitian=True)


def Z(*qubits):
    return Operator("Z", qubits, hermitian=True)


# --------------------------------------------------------------------------------
# Hadamard gate
# --------------------------------------------------------------------------------


def H(*qubits):
    return Operator("H", qubits, hermitian=True)


# --------------------------------------------------------------------------------
# Control gates
# --------------------------------------------------------------------------------


def CX(control, target):
    if control == target:
        raise ValueError("Control and target must differ.")
    return Operator("CX", (control, target), hermitian=True)


def CY(control, target):
    if control == target:
        raise ValueError("Control and target must differ.")
    return Operator("CY", (control, target), hermitian=True)


def CZ(control, target):
    if control == target:
        raise ValueError("Control and target must differ.")
    return Operator("CZ", (control, target), hermitian=True)


def CCX(control1, control2, target):
    if control1 == control2 or control1 == target or control2 == target:
        raise ValueError("All qubits must be distinct.")
    return Operator("CCX", (control1, control2, target), hermitian=True)


def CCY(control1, control2, target):
    if control1 == control2 or control1 == target or control2 == target:
        raise ValueError("All qubits must be distinct.")
    return Operator("CCY", (control1, control2, target), hermitian=True)


def CCZ(control1, control2, target):
    if control1 == control2 or control1 == target or control2 == target:
        raise ValueError("All qubits must be distinct.")
    return Operator("CCZ", (control1, control2, target), hermitian=True)


# --------------------------------------------------------------------------------
# Phase gates
# --------------------------------------------------------------------------------


def S(*qubits):
    return Operator("S", qubits, inverse_operator=Sdg)


def Sdg(*qubits):
    return Operator("Sdg", qubits, inverse_operator=S)


def T(*qubits):
    return Operator("T", qubits, inverse_operator=Tdg)


def Tdg(*qubits):
    return Operator("Tdg", qubits, inverse_operator=T)


# --------------------------------------------------------------------------------
# Rotation gates
# --------------------------------------------------------------------------------


def RX(theta):
    parameter = theta if isinstance(theta, Parameter) else Parameter(theta)
    return lambda *qubits: Operator("RX", qubits, (parameter,), inverse_operator=lambda *q: RX(-theta)(*q))


def RY(theta):
    parameter = theta if isinstance(theta, Parameter) else Parameter(theta)
    return lambda *qubits: Operator("RY", qubits, (parameter,), inverse_operator=lambda *q: RY(-theta)(*q))


def RZ(theta):
    parameter = theta if isinstance(theta, Parameter) else Parameter(theta)
    return lambda *qubits: Operator("RZ", qubits, (parameter,), inverse_operator=lambda *q: RZ(-theta)(*q))
