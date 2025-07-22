import jax.numpy as jnp
import pytest
from qmlir.operator import CX, H, X, Z
from qmlir.runtime.jax.simulator import JaxSimulator
from qmlir.circuit import QuantumCircuit


def test_expectation_hamiltonian_on_bell_state():
    """Test expectation of H = 0.5 * Z(0)@Z(1) + 0.5 * X(0) on a Bell state."""
    simulator = JaxSimulator()
    circuit = QuantumCircuit(2)
    with circuit:
        H(0)
        CX(0, 1)

    # Define Hamiltonian observable
    observable = 0.5 * (Z(0) @ Z(1)) + 0.5 * X(0)

    value = simulator.expectation(circuit, observable)

    print()
    print(f"Expectation value: {value}")

    assert abs(value - 0.5) < 1e-6, f"Expected 0.5, got {value}"


def test_simulator_measurement_bitstring_endianness():
    """Check simulator outputs correctly formatted bitstrings based on circuit.little_endian flag."""
    # Circuit to prepare |011⟩ (q2=0, q1=1, q0=1)
    shots = 100
    for little_endian in (True, False):
        circuit = QuantumCircuit(3, little_endian=little_endian)
        with circuit:
            X(0)  # flip q0
            X(1)  # flip q1

        sim = JaxSimulator()
        samples = sim.measure(circuit, shots)

        assert len(samples) == 1, f"Expected single bitstring, got {samples}"
        ((bitstring, count),) = samples.items()
        assert count == shots, f"Expected {shots} shots for single state, got {count}"

        if little_endian:
            # q2=0, q1=1, q0=1 → '011'
            expected = "011"
        else:
            # q0=1, q1=1, q2=0 → '110'
            expected = "110"

        assert bitstring == expected, (
            f"Endianness: {'little' if little_endian else 'big'} — Expected {expected}, got {bitstring}"
        )


def test_observable_on_011_state():
    """Test Z(0), Z(1), Z(0,1) on |011⟩ = X(0); X(1) under little-endian."""
    sim = JaxSimulator()
    circuit = QuantumCircuit(3)
    with circuit:
        X(0)  # flip q0 (LSB)
        X(1)  # flip q1

    # Verify statevector: |011⟩ should have index 3 = 0b011
    sv = sim.statevector(circuit)
    assert jnp.argmax(jnp.abs(sv)) == 3, "Expected state to be |011⟩ (index 3)"

    # Z(0): q0 = 1 → eigenvalue -1
    print("Expectation value of Z(0):", sim.expectation(circuit, Z(0)))
    assert abs(sim.expectation(circuit, Z(0)) + 1.0) < 1e-6

    # Z(1): q1 = 1 → eigenvalue -1
    print("Expectation value of Z(1):", sim.expectation(circuit, Z(1)))
    assert abs(sim.expectation(circuit, Z(1)) + 1.0) < 1e-6

    # Z(0, 1): (-1) * (-1) = +1
    print("Expectation value of Z(0, 1):", sim.expectation(circuit, Z(0, 1)))
    assert abs(sim.expectation(circuit, Z(0, 1)) - 1.0) < 1e-6

    # Measurement: should be '011'
    samples = sim.measure(circuit, shots=100)
    assert set(samples.keys()) == {"011"}, f"Expected bitstring '011', got {samples}"


def test_tensor_product_duplicate_qubit_raises():
    """Ensure tensor product with repeated qubit index raises ValueError."""
    with pytest.raises(ValueError, match="Duplicate qubit"):
        _ = X(0) @ X(0)
