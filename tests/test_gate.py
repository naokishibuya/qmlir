import pytest
from typing import Tuple
from qmlir.circuit import QuantumCircuit
from qmlir.gate import Gate
from qmlir.operator import I, X, Y, Z, H, CX, CY, CZ, S, Sdg, T, Tdg, RX, RY, RZ
from qmlir.parameter import Parameter


# Categorize the gates
PAULI_GATES = [I, X, Y, Z]
HADMARD_GATES = [H]
PHASE_GATES = [S, Sdg, T, Tdg]
TWO_QUBIT_GATES = [CX, CY, CZ]
ROTATION_GATES = [RX, RY, RZ]


@pytest.fixture
def circuit():
    return QuantumCircuit(2)


@pytest.mark.parametrize("gate", PAULI_GATES + HADMARD_GATES)
def test_pauli_gate(gate, circuit):
    with circuit:
        inst = gate(0)
    assert circuit.num_qubits == 2
    assert len(inst.qubits) == 1
    assert inst.name in repr(inst)


@pytest.mark.parametrize("gate", PHASE_GATES)
def test_phase_gate(gate, circuit):
    with circuit:
        inst = gate(0)
    assert circuit.num_qubits == 2
    assert len(inst.qubits) == 1
    assert inst.name in repr(inst)


@pytest.mark.parametrize("gate", TWO_QUBIT_GATES)
def test_two_qubit_gate(gate, circuit):
    with circuit:
        inst = gate(0, 1)
    assert circuit.num_qubits == 2
    assert len(inst.qubits) == 2
    assert inst.name in repr(inst)


@pytest.mark.parametrize("gate", ROTATION_GATES)
def test_rotation_gate(gate, circuit):
    with circuit:
        theta = 1.23
        inst = gate(theta)(0)
    assert circuit.num_qubits == 2
    assert len(inst.qubits) == 1
    assert inst.name in repr(inst)
    assert inst.parameters[0].value == theta


def test_unknown_gate_raises():
    class DummyGate:
        def __init__(
            self,
            name: str,
            qubits: Tuple[int, ...],
            parameters: Tuple[Parameter, ...] = (),
        ):
            self.name = name
            self.qubits = qubits
            self.parameters = parameters

    qc = QuantumCircuit(1)
    with pytest.raises(TypeError, match="Expected Gate, got DummyGate"):
        qc.append(DummyGate("X", (0, 1)))


def test_rotation_operator_construction():
    with QuantumCircuit(1):
        theta = Parameter(1.23)
        obs = RX(theta)(0)
    assert isinstance(obs, Gate)
    assert obs.name == "RX"
    assert obs.parameters[0].value == 1.23


def test_parameter_negation():
    with QuantumCircuit(1):
        p = Parameter(0.7, name="theta")
        neg = -p
    assert neg.value == -0.7
    assert neg.name.startswith("-theta")


def test_operator_adds_to_active_circuit():
    c = QuantumCircuit(2)
    with c:
        X(0)
        Y(1)
    assert len(c.gates) == 2
    assert c.gates[0].name == "X"
    assert c.gates[1].name == "Y"


def test_operator_qubit_range_check():
    circuit = QuantumCircuit(2)
    with pytest.raises(ValueError):
        with circuit:
            X(2)  # Invalid qubit index


def test_rotation_gate_factory_lambda():
    with QuantumCircuit(2):
        rx_obs = RX(0.3)
        gate = rx_obs(1)
    assert gate.name == "RX"
    assert gate.qubits == (1,)
    assert gate.parameters[0].value == 0.3
