from .circuit import QuantumCircuit
from .gate import Gate
from .observable import Observable
from .parameter import Parameter


# --------------------------------------------------------------------------------
# Pauli operators
# --------------------------------------------------------------------------------


def I(*qubits):  # noqa: E743
    if QuantumCircuit.current():
        return Gate("I", qubits, hermitian=True, self_inverse=True)
    return Observable("I", qubits)


def X(*qubits):
    if QuantumCircuit.current():
        return Gate("X", qubits, hermitian=True, self_inverse=True)
    return Observable("X", qubits)


def Y(*qubits):
    if QuantumCircuit.current():
        return Gate("Y", qubits, hermitian=True, self_inverse=True)
    return Observable("Y", qubits)


def Z(*qubits):
    if QuantumCircuit.current():
        return Gate("Z", qubits, hermitian=True, self_inverse=True)
    return Observable("Z", qubits)


# --------------------------------------------------------------------------------
# Hadamard gate
# --------------------------------------------------------------------------------


def H(*qubits):
    return Gate("H", qubits, hermitian=True, self_inverse=True)


# --------------------------------------------------------------------------------
# Control gates
# --------------------------------------------------------------------------------


def CX(control, target):
    if control == target:
        raise ValueError("Control and target must differ.")
    return Gate("CX", (control, target), hermitian=True, self_inverse=True)


def CY(control, target):
    if control == target:
        raise ValueError("Control and target must differ.")
    return Gate("CY", (control, target), hermitian=True, self_inverse=True)


def CZ(control, target):
    if control == target:
        raise ValueError("Control and target must differ.")
    return Gate("CZ", (control, target), hermitian=True, self_inverse=True)


def CCX(control1, control2, target):
    if control1 == control2 or control1 == target or control2 == target:
        raise ValueError("All qubits must be distinct.")
    return Gate("CCX", (control1, control2, target), hermitian=True, self_inverse=True)


def CCY(control1, control2, target):
    if control1 == control2 or control1 == target or control2 == target:
        raise ValueError("All qubits must be distinct.")
    return Gate("CCY", (control1, control2, target), hermitian=True, self_inverse=True)


def CCZ(control1, control2, target):
    if control1 == control2 or control1 == target or control2 == target:
        raise ValueError("All qubits must be distinct.")
    return Gate("CCZ", (control1, control2, target), hermitian=True, self_inverse=True)


# --------------------------------------------------------------------------------
# Phase gates
# --------------------------------------------------------------------------------


def S(*qubits):
    return Gate("S", qubits)


def Sdg(*qubits):
    return Gate("Sdg", qubits)


def T(*qubits):
    return Gate("T", qubits)


def Tdg(*qubits):
    return Gate("Tdg", qubits)


# --------------------------------------------------------------------------------
# Rotation gates
# --------------------------------------------------------------------------------


def RX(theta):
    parameter = theta if isinstance(theta, Parameter) else Parameter(theta)
    return lambda *qubits: Gate("RX", qubits, (parameter,))


def RY(theta):
    parameter = theta if isinstance(theta, Parameter) else Parameter(theta)
    return lambda *qubits: Gate("RY", qubits, (parameter,))


def RZ(theta):
    parameter = theta if isinstance(theta, Parameter) else Parameter(theta)
    return lambda *qubits: Gate("RZ", qubits, (parameter,))
