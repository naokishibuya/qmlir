from .circuit import QuantumCircuit
from .gate import Gate
from .observable import Observable
from .parameter import Parameter


# --------------------------------------------------------------------------------
# Pauli operators (Gate if in circuit, Observable otherwise)
# --------------------------------------------------------------------------------


def I(*qubits):  # noqa: E743
    return Gate("I", qubits) if QuantumCircuit.current() else Observable("I", qubits)


def X(*qubits):
    return Gate("X", qubits) if QuantumCircuit.current() else Observable("X", qubits)


def Y(*qubits):
    return Gate("Y", qubits) if QuantumCircuit.current() else Observable("Y", qubits)


def Z(*qubits):
    return Gate("Z", qubits) if QuantumCircuit.current() else Observable("Z", qubits)


# --------------------------------------------------------------------------------
# Hadamard gate
# --------------------------------------------------------------------------------


def H(*qubits):
    return Gate("H", qubits)


# --------------------------------------------------------------------------------
# Control gates
# --------------------------------------------------------------------------------


def CX(control, target):
    if control == target:
        raise ValueError("Control and target must differ.")
    return Gate("CX", (control, target))


def CY(control, target):
    if control == target:
        raise ValueError("Control and target must differ.")
    return Gate("CY", (control, target))


def CZ(control, target):
    if control == target:
        raise ValueError("Control and target must differ.")
    return Gate("CZ", (control, target))


def CCX(control1, control2, target):
    if control1 == control2 or control1 == target or control2 == target:
        raise ValueError("All qubits must be distinct.")
    return Gate("CCX", (control1, control2, target))


def CCY(control1, control2, target):
    if control1 == control2 or control1 == target or control2 == target:
        raise ValueError("All qubits must be distinct.")
    return Gate("CCY", (control1, control2, target))


def CCZ(control1, control2, target):
    if control1 == control2 or control1 == target or control2 == target:
        raise ValueError("All qubits must be distinct.")
    return Gate("CCZ", (control1, control2, target))


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
