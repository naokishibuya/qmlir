import pytest
from qmlir.circuit import QuantumCircuit
from qmlir.operator import I, X, Y, Z, H, CX, CY, CZ, S, Sdg, T, Tdg, RX, RY, RZ

# Categorize the operators
SINGLE_QUBIT = [I, X, Y, Z, H, S, Sdg, T, Tdg]
TWO_QUBIT = [CX, CY, CZ]
ROTATIONS = [RX, RY, RZ]


@pytest.fixture
def circuit():
    return QuantumCircuit(2)


@pytest.fixture
def qubits(circuit):
    return circuit.qubits[0], circuit.qubits[1]


@pytest.mark.parametrize("op", SINGLE_QUBIT)
def test_single_qubit_operator(op, qubits):
    q0, _ = qubits
    inst = op(q0)
    assert inst.qubits == (q0,)
    assert isinstance(repr(inst), str)


@pytest.mark.parametrize("op", TWO_QUBIT)
def test_two_qubit_operator(op, qubits):
    q0, q1 = qubits
    inst = op(q0, q1)
    assert inst.qubits == (q0, q1)
    assert isinstance(repr(inst), str)


@pytest.mark.parametrize("op", ROTATIONS)
def test_rotation_operator(op, qubits):
    q0, _ = qubits
    theta = 1.23
    inst = op(theta)(q0)
    assert inst.qubits == (q0,)
    assert inst.parameters[0].value == theta
    assert isinstance(repr(inst), str)


@pytest.mark.parametrize("op", [S, Sdg, T, Tdg, RX, RY, RZ])
def test_inverse_exists(op, qubits):
    q0, _ = qubits
    args = (1.23, q0) if op in ROTATIONS else (q0,)
    inst = op(args[0])(args[1]) if len(args) == 2 else op(args[0])
    inv = inst.inverse()
    assert inv is not None
    assert inv.qubits == inst.qubits
    if inst in (RX, RY, RZ):
        assert inv.parameters[0] == -inst.parameters[0]


@pytest.mark.parametrize("op", [I, X, Y, Z, H, CX, CY, CZ])
def test_self_inverse(op, qubits):
    q0, q1 = qubits
    args = (q0, q1) if op in TWO_QUBIT else (q0,)
    inst = op(*args)
    assert inst.inverse() == inst
