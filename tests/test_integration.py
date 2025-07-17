"""Integration tests with quantum-opt tool."""

import subprocess
from qmlir import Circuit, circuit_to_mlir
from qmlir.config import get_quantum_opt_path


class TestQuantumOptIntegration:
    """Test integration with quantum-opt tool."""

    def test_quantum_opt_available(self):
        """Test that quantum-opt is available and working."""
        try:
            quantum_opt_path = get_quantum_opt_path()
            result = subprocess.run([quantum_opt_path, "--help"], capture_output=True, text=True, timeout=10)
            assert result.returncode == 0
            assert "quantum-opt" in result.stdout or "OVERVIEW" in result.stdout
        except (FileNotFoundError, RuntimeError):
            pytest.skip("quantum-opt not available")

    def test_bell_state_with_quantum_opt(self):
        """Test Bell state MLIR with quantum-opt."""
        circuit = Circuit()
        circuit.h(0).cx(0, 1)
        mlir_code = circuit_to_mlir(circuit, "bell_state")

        try:
            quantum_opt_path = get_quantum_opt_path()
            result = subprocess.run([quantum_opt_path], input=mlir_code, capture_output=True, text=True, timeout=10)

            assert result.returncode == 0
            assert "module {" in result.stdout
            assert "func.func @bell_state()" in result.stdout
            assert '"quantum.alloc"()' in result.stdout
            assert '"quantum.h"(' in result.stdout
            assert '"quantum.cx"(' in result.stdout

        except (FileNotFoundError, RuntimeError):
            pytest.skip("quantum-opt not available")

    def test_double_x_cancellation(self):
        """Test double X cancellation with quantum-opt."""
        circuit = Circuit()
        circuit.x(0).x(0)
        mlir_code = circuit_to_mlir(circuit, "double_x_test")

        try:
            quantum_opt_path = get_quantum_opt_path()
            result = subprocess.run(
                [quantum_opt_path, "--quantum-cancel-x"], input=mlir_code, capture_output=True, text=True, timeout=10
            )

            assert result.returncode == 0
            assert "module {" in result.stdout
            assert "func.func @double_x_test()" in result.stdout

            # After cancellation, should have no X gates
            assert '"quantum.x"(' not in result.stdout

            # Should still have qubit allocation
            assert '"quantum.alloc"()' in result.stdout

        except (FileNotFoundError, RuntimeError):
            pytest.skip("quantum-opt not available")

    def test_complex_circuit_with_quantum_opt(self):
        """Test complex circuit with quantum-opt."""
        circuit = Circuit()
        circuit.h(0).h(1).x(2).cx(0, 1).cx(1, 2)
        mlir_code = circuit_to_mlir(circuit, "complex_circuit")

        try:
            quantum_opt_path = get_quantum_opt_path()
            result = subprocess.run([quantum_opt_path], input=mlir_code, capture_output=True, text=True, timeout=10)

            assert result.returncode == 0
            assert "module {" in result.stdout
            assert "func.func @complex_circuit()" in result.stdout

            # Should preserve all gates (no cancellation without passes)
            assert '"quantum.h"(' in result.stdout
            assert '"quantum.x"(' in result.stdout
            assert '"quantum.cx"(' in result.stdout

        except (FileNotFoundError, RuntimeError):
            pytest.skip("quantum-opt not available")


# Skip integration tests if pytest is not available
try:
    import pytest
except ImportError:
    pytest = None

    class TestQuantumOptIntegration:
        """Dummy class when pytest is not available."""

        pass
