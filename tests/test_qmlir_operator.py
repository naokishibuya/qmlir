import pytest
from qmlir.circuit import QuantumCircuit
from qmlir.operator import I, X, Y, Z, H, CX, CY, CZ, S, Sdg, T, Tdg, RX, RY, RZ

# Categorize the operators
PAULI_OPERATORS = [I, X, Y, Z]
HADMARD_OPERATORS = [H]
PHASE_OPERATORS = [S, Sdg, T, Tdg]
TWO_QUBIT_OPERATORS = [CX, CY, CZ]
ROTATION_OPERATORS = [RX, RY, RZ]


@pytest.fixture
def circuit():
    return QuantumCircuit(2)


@pytest.mark.parametrize("op", PAULI_OPERATORS + HADMARD_OPERATORS)
def test_pauli_operator(op, circuit):
    with circuit:
        inst = op(0)
    assert circuit.num_qubits == 2
    assert len(inst.qubits) == 1
    assert inst.name in repr(inst)
    assert inst.unitary
    assert inst.hermitian
    assert inst.self_inverse


@pytest.mark.parametrize("op", PHASE_OPERATORS)
def test_phase_operator(op, circuit):
    with circuit:
        inst = op(0)
    assert circuit.num_qubits == 2
    assert len(inst.qubits) == 1
    assert inst.name in repr(inst)
    assert inst.unitary
    assert inst.hermitian is False
    assert inst.self_inverse is False


@pytest.mark.parametrize("op", TWO_QUBIT_OPERATORS)
def test_two_qubit_operator(op, circuit):
    with circuit:
        inst = op(0, 1)
    assert circuit.num_qubits == 2
    assert len(inst.qubits) == 2
    assert inst.name in repr(inst)
    assert inst.unitary
    assert inst.hermitian is True
    assert inst.self_inverse is True


@pytest.mark.parametrize("op", ROTATION_OPERATORS)
def test_rotation_operator(op, circuit):
    with circuit:
        theta = 1.23
        inst = op(theta)(0)
    assert circuit.num_qubits == 2
    assert len(inst.qubits) == 1
    assert inst.name in repr(inst)
    assert inst.parameters[0].value == theta
    assert inst.unitary
    assert inst.hermitian is False
    assert inst.self_inverse is False
