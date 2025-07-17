"""Test Phase 2 parametric circuit implementation.

Tests the full C++ MLIR backend support for parametric gates including:
- MLIR dialect recognition of parametric gates
- Optimization passes with parametric gates
- End-to-end transpilation with parametric circuits
"""

from qmlir import QuantumCircuit, Parameter, transpile
from qmlir.compiler import circuit_to_mlir


class TestParametricMlirDialect:
    """Test that the MLIR dialect properly handles parametric gates."""

    def test_rx_gate_mlir_generation(self):
        """Test RX gate MLIR generation."""
        circuit = QuantumCircuit(1)
        theta = Parameter(0.5, "theta")
        circuit.rx(0, theta)

        mlir_code = circuit_to_mlir(circuit)
        assert '"quantum.rx"' in mlir_code
        assert "(i32, f64) -> ()" in mlir_code
        assert "%arg0: f64" in mlir_code

    def test_ry_gate_mlir_generation(self):
        """Test RY gate MLIR generation."""
        circuit = QuantumCircuit(1)
        phi = Parameter(1.0, "phi")
        circuit.ry(0, phi)

        mlir_code = circuit_to_mlir(circuit)
        assert '"quantum.ry"' in mlir_code
        assert "(i32, f64) -> ()" in mlir_code

    def test_rz_gate_mlir_generation(self):
        """Test RZ gate MLIR generation."""
        circuit = QuantumCircuit(1)
        alpha = Parameter(2.0, "alpha")
        circuit.rz(0, alpha)

        mlir_code = circuit_to_mlir(circuit)
        assert '"quantum.rz"' in mlir_code
        assert "(i32, f64) -> ()" in mlir_code

    def test_all_rotation_gates_together(self):
        """Test all rotation gates in one circuit."""
        circuit = QuantumCircuit(1)
        theta = Parameter(0.5, "theta")
        phi = Parameter(1.0, "phi")
        alpha = Parameter(1.5, "alpha")

        circuit.rx(0, theta)
        circuit.ry(0, phi)
        circuit.rz(0, alpha)

        mlir_code = circuit_to_mlir(circuit)
        assert '"quantum.rx"' in mlir_code
        assert '"quantum.ry"' in mlir_code
        assert '"quantum.rz"' in mlir_code
        assert "%arg0: f64, %arg1: f64, %arg2: f64" in mlir_code


class TestParametricOptimization:
    """Test optimization passes with parametric gates."""

    def test_rotation_gates_do_not_self_cancel(self):
        """Test that rotation gates don't self-cancel like Pauli gates."""
        circuit = QuantumCircuit(1)
        theta = Parameter(0.5, "theta")

        # Add two RX gates with same parameter - they should NOT cancel
        circuit.rx(0, theta)
        circuit.rx(0, theta)

        # Add two X gates - they SHOULD cancel
        circuit.x(0)
        circuit.x(0)

        unoptimized = transpile(circuit, optimization_level=0)
        optimized = transpile(circuit, optimization_level=1)

        # RX gates should remain (they combine, don't cancel)
        assert unoptimized.count('"quantum.rx"') == 2
        assert optimized.count('"quantum.rx"') == 2

        # X gates should be removed by optimization
        assert unoptimized.count('"quantum.x"') == 2
        assert optimized.count('"quantum.x"') == 0

    def test_mixed_parametric_and_fixed_gates_optimization(self):
        """Test optimization with mix of parametric and fixed gates."""
        circuit = QuantumCircuit(2)
        theta = Parameter(0.5, "theta")

        circuit.h(0)
        circuit.h(0)  # Should cancel
        circuit.ry(1, theta)
        circuit.cx(0, 1)
        circuit.ry(1, theta)  # Should remain (doesn't cancel)
        circuit.z(0)
        circuit.z(0)  # Should cancel

        unoptimized = transpile(circuit, optimization_level=0)
        optimized = transpile(circuit, optimization_level=1)

        # H gates should cancel
        assert unoptimized.count('"quantum.h"') == 2
        assert optimized.count('"quantum.h"') == 0

        # Z gates should cancel
        assert unoptimized.count('"quantum.z"') == 2
        assert optimized.count('"quantum.z"') == 0

        # RY gates should remain
        assert unoptimized.count('"quantum.ry"') == 2
        assert optimized.count('"quantum.ry"') == 2

        # CX should remain
        assert unoptimized.count('"quantum.cx"') == 1
        assert optimized.count('"quantum.cx"') == 1


class TestParametricTranspilation:
    """Test end-to-end transpilation with parametric circuits."""

    def test_vqe_ansatz_transpilation(self):
        """Test transpilation of a VQE-style ansatz."""
        circuit = QuantumCircuit(2)
        theta = Parameter(0.5, "theta")
        phi = Parameter(1.0, "phi")

        # Simple ansatz
        circuit.ry(0, theta)
        circuit.ry(1, phi)
        circuit.cx(0, 1)
        circuit.ry(0, theta)  # Reuse parameter

        mlir_code = transpile(circuit)

        # Check function signature has correct parameters
        assert "@main(%arg0: f64, %arg1: f64)" in mlir_code

        # Check all gates are present
        assert mlir_code.count('"quantum.ry"') == 3
        assert '"quantum.cx"' in mlir_code

        # Check parameter reuse (theta appears twice)
        assert mlir_code.count("%arg0") >= 2

    def test_parameter_ordering_consistency(self):
        """Test that parameter ordering is consistent."""
        circuit = QuantumCircuit(3)
        alpha = Parameter(1.0, "alpha")
        beta = Parameter(2.0, "beta")
        gamma = Parameter(3.0, "gamma")

        # Use parameters in specific order
        circuit.rx(0, beta)  # Second parameter first
        circuit.ry(1, alpha)  # First parameter second
        circuit.rz(2, gamma)  # Third parameter third
        circuit.rx(0, alpha)  # Reuse first parameter

        mlir_code = circuit_to_mlir(circuit)

        # Should have 3 parameters in function signature
        assert "%arg0: f64, %arg1: f64, %arg2: f64" in mlir_code

        # Parameters should be consistently mapped
        lines = mlir_code.split("\n")
        rx_lines = [line for line in lines if '"quantum.rx"' in line]
        ry_lines = [line for line in lines if '"quantum.ry"' in line]
        rz_lines = [line for line in lines if '"quantum.rz"' in line]

        # First RX uses beta (should be first parameter encountered)
        assert "%arg0" in rx_lines[0]
        # RY uses alpha (should be second parameter encountered)
        assert "%arg1" in ry_lines[0]
        # RZ uses gamma (should be third parameter encountered)
        assert "%arg2" in rz_lines[0]
        # Second RX reuses alpha (should use same arg as RY)
        assert "%arg1" in rx_lines[1]

    def test_large_parametric_circuit(self):
        """Test transpilation of a large parametric circuit."""
        n_qubits = 4
        n_layers = 3
        circuit = QuantumCircuit(n_qubits)

        parameters = []
        for layer in range(n_layers):
            for qubit in range(n_qubits):
                param = Parameter(0.1 * (layer * n_qubits + qubit), f"theta_{layer}_{qubit}")
                parameters.append(param)
                circuit.ry(qubit, param)

            # Add entangling gates
            for qubit in range(n_qubits - 1):
                circuit.cx(qubit, qubit + 1)

        mlir_code = transpile(circuit)

        # Should have correct number of parameters
        expected_params = n_layers * n_qubits
        param_signature = f"@main(%{', %'.join([f'arg{i}: f64' for i in range(expected_params)])})"
        assert param_signature in mlir_code

        # Should have correct number of gates
        expected_ry_gates = n_layers * n_qubits
        expected_cx_gates = n_layers * (n_qubits - 1)

        assert mlir_code.count('"quantum.ry"') == expected_ry_gates
        assert mlir_code.count('"quantum.cx"') == expected_cx_gates
