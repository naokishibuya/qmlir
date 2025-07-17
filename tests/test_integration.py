"""Integration tests with quantum-opt tool."""

from qmlir import QuantumCircuit, circuit_to_mlir, run_quantum_optimizer


class TestQuantumOptIntegration:
    """Test integration with quantum-opt tool."""

    def test_quantum_opt_available(self):
        """Test that quantum-opt is available and working."""
        result = run_quantum_optimizer("", "--help")
        assert result.returncode == 0
        assert "quantum-opt" in result.stdout or "OVERVIEW" in result.stdout

    def test_bell_state_with_quantum_opt(self):
        """Test Bell state MLIR with quantum-opt."""
        circuit = QuantumCircuit(2)
        circuit.h(0).cx(0, 1)
        mlir_code = circuit_to_mlir(circuit, "bell_state")

        result = run_quantum_optimizer(mlir_code)

        assert result.returncode == 0
        assert "module {" in result.stdout
        assert "func.func @bell_state()" in result.stdout
        assert '"quantum.alloc"()' in result.stdout
        assert '"quantum.h"(' in result.stdout
        assert '"quantum.cx"(' in result.stdout

    def test_double_x_cancellation(self):
        """Test double X cancellation with quantum-opt."""
        circuit = QuantumCircuit(1)
        circuit.x(0).x(0)
        mlir_code = circuit_to_mlir(circuit, "double_x_test")

        result = run_quantum_optimizer(mlir_code, "--quantum-cancel-self-inverse")

        assert result.returncode == 0
        assert "module {" in result.stdout
        assert "func.func @double_x_test()" in result.stdout

        # After cancellation, should have no X gates
        assert '"quantum.x"(' not in result.stdout

        # Should still have qubit allocation
        assert '"quantum.alloc"()' in result.stdout

    def test_complex_circuit_with_quantum_opt(self):
        """Test complex circuit with quantum-opt."""
        circuit = QuantumCircuit(3)
        circuit.h(0).h(1).x(2).cx(0, 1).cx(1, 2)
        mlir_code = circuit_to_mlir(circuit, "complex_circuit")

        result = run_quantum_optimizer(mlir_code)

        assert result.returncode == 0
        assert "module {" in result.stdout
        assert "func.func @complex_circuit()" in result.stdout

        # Should preserve all gates (no cancellation without passes)
        assert '"quantum.h"(' in result.stdout
        assert '"quantum.x"(' in result.stdout
        assert '"quantum.cx"(' in result.stdout
