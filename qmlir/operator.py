from typing import Tuple
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
        qstr = ", ".join(f"q{q}" for q in self.qubits)
        pstr = f"({', '.join(map(str, self.parameters))})" if self.parameters else ""
        return f"{self.name}{pstr}|{qstr}⟩"


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
