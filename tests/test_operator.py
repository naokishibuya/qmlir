import numpy as np
from qmlir.operator import X, Y, H, CX, CY, CZ
from qmlir.circuit import QuantumCircuit
from qmlir.runtime import JaxSimulator


def test_cx_operator():
    circuit = QuantumCircuit(2)
    with circuit:
        H(0)
        CX(0, 1)
    simulator = JaxSimulator()
    probs = simulator.probabilities(circuit)
    expected_probs = [0.5, 0.0, 0.0, 0.5]
    assert np.allclose(probs, expected_probs, atol=1e-6), f"Expected {expected_probs}, got {probs}"


def test_cy_operator():
    circuit = QuantumCircuit(2)
    with circuit:
        Y(0)
        CY(0, 1)
    simulator = JaxSimulator()
    probs = simulator.probabilities(circuit)
    expected_probs = [0.0, 0.0, 0.0, 1.0]
    assert np.allclose(probs, expected_probs, atol=1e-6), f"Expected {expected_probs}, got {probs}"


def test_cz_operator():
    circuit = QuantumCircuit(3)
    with circuit:
        X(0)
        CZ(0, 1)
    simulator = JaxSimulator()
    probs = simulator.probabilities(circuit)
    expected_probs = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert np.allclose(probs, expected_probs, atol=1e-6), f"Expected {expected_probs}, got {probs}"


def test_cx_middle_to_lsb():
    circuit = QuantumCircuit(3)
    with circuit:
        X(0)  # q0 = 1
        X(1)  # q1 = 1 (control)
        CX(1, 2)  # should flip q2
    sim = JaxSimulator()
    probs = sim.probabilities(circuit)
    expected = [0.0] * 2**3
    expected[7] = 1.0  # |111⟩ state => |7⟩
    assert np.allclose(probs, expected, atol=1e-6)
