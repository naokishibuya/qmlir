"""Tests for qmlir.mlir.transpiler module."""

import pytest
from qmlir.mlir.transpiler import circuit_to_mlir, apply_passes
from qmlir.circuit import QuantumCircuit
from qmlir.parameter import Parameter


class TestCircuitToMLIR:
    """Test circuit_to_mlir function."""

    def test_empty_circuit(self):
        """Test transpilation of empty circuit."""
        circuit = QuantumCircuit(2)
        mlir = circuit_to_mlir(circuit)
        assert "module" in mlir
        assert "func.func @main()" in mlir
        assert "quantum.alloc" in mlir

    def test_single_qubit_gate(self):
        """Test transpilation of single-qubit gate."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        mlir = circuit_to_mlir(circuit)
        assert "quantum.x" in mlir
        assert "quantum.alloc" in mlir

    def test_two_qubit_gate(self):
        """Test transpilation of two-qubit gate."""
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        mlir = circuit_to_mlir(circuit)
        assert "quantum.cx" in mlir
        assert "quantum.alloc" in mlir

    def test_parametric_gate(self):
        """Test transpilation of parametric gate."""
        circuit = QuantumCircuit(1)
        param = Parameter(0.5, name="theta")
        circuit.rx(0, param)
        mlir = circuit_to_mlir(circuit)
        assert "quantum.rx" in mlir
        assert "func.func @main(%arg0: f64)" in mlir

    def test_multiple_gates(self):
        """Test transpilation of multiple gates."""
        circuit = QuantumCircuit(2)
        circuit.h(0).cx(0, 1).x(1)
        mlir = circuit_to_mlir(circuit)
        assert "quantum.h" in mlir
        assert "quantum.cx" in mlir
        assert "quantum.x" in mlir

    def test_bell_state_circuit(self):
        """Test transpilation of Bell state circuit."""
        circuit = QuantumCircuit(2)
        circuit.h(0).cx(0, 1)
        mlir = circuit_to_mlir(circuit)
        assert "quantum.h" in mlir
        assert "quantum.cx" in mlir
        assert "return" in mlir

    def test_custom_function_name(self):
        """Test transpilation with custom function name."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        mlir = circuit_to_mlir(circuit, function_name="custom_func")
        assert "func.func @custom_func()" in mlir

    def test_all_gate_types(self):
        """Test transpilation of all gate types."""
        circuit = QuantumCircuit(2)
        param = Parameter(0.5, name="theta")

        # Add one of each gate type
        circuit.i(0).x(1).y(0).z(1).h(0).s(1).t(0).sdg(1).tdg(0)
        circuit.cx(0, 1).cy(1, 0).cz(0, 1)
        circuit.rx(0, param).ry(1, param).rz(0, param)

        mlir = circuit_to_mlir(circuit)

        # Check that all gate types are present
        gate_types = [
            "quantum.i",
            "quantum.x",
            "quantum.y",
            "quantum.z",
            "quantum.h",
            "quantum.s",
            "quantum.t",
            "quantum.sdg",
            "quantum.tdg",
            "quantum.cx",
            "quantum.cy",
            "quantum.cz",
            "quantum.rx",
            "quantum.ry",
            "quantum.rz",
        ]

        for gate_type in gate_types:
            assert gate_type in mlir


class TestApplyPasses:
    """Test apply_passes function."""

    def test_basic_optimization(self):
        """Test basic optimization pass."""
        # Create MLIR with self-inverse operations
        circuit = QuantumCircuit(2)
        circuit.x(0).x(0)  # X*X = I
        mlir = circuit_to_mlir(circuit)

        # Apply optimization
        optimized = apply_passes(mlir)
        assert "module" in optimized
        assert "func.func @main()" in optimized

    def test_optimization_with_custom_args(self):
        """Test optimization with custom arguments."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        mlir = circuit_to_mlir(circuit)

        # Apply optimization with custom pass
        optimized = apply_passes(mlir, "--quantum-cancel-self-inverse")
        assert "module" in optimized

    def test_optimization_timeout(self):
        """Test optimization timeout handling."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        mlir = circuit_to_mlir(circuit)

        # Apply optimization with short timeout
        optimized = apply_passes(mlir, timeout=1)
        assert "module" in optimized

    def test_optimization_error_handling(self):
        """Test optimization error handling."""
        invalid_mlir = "invalid mlir code"

        with pytest.raises(RuntimeError):
            apply_passes(invalid_mlir)

    def test_optimization_preserves_structure(self):
        """Test that optimization preserves MLIR structure."""
        circuit = QuantumCircuit(2)
        circuit.h(0).cx(0, 1)
        mlir = circuit_to_mlir(circuit)

        optimized = apply_passes(mlir)

        # Should still be valid MLIR
        assert "module" in optimized
        assert "func.func @main()" in optimized
        assert "return" in optimized


class TestTranspilerIntegration:
    """Test integration between transpilation and optimization."""

    def test_full_pipeline(self):
        """Test full transpilation and optimization pipeline."""
        circuit = QuantumCircuit(2)
        circuit.h(0).cx(0, 1)

        # Transpile
        mlir = circuit_to_mlir(circuit)
        assert "quantum.h" in mlir
        assert "quantum.cx" in mlir

        # Optimize
        optimized = apply_passes(mlir)
        assert "module" in optimized

    def test_parametric_pipeline(self):
        """Test full pipeline with parametric gates."""
        circuit = QuantumCircuit(1)
        param = Parameter(0.5, name="theta")
        circuit.rx(0, param)

        # Transpile
        mlir = circuit_to_mlir(circuit)
        assert "quantum.rx" in mlir
        assert "f64" in mlir  # Parameter type

        # Optimize
        optimized = apply_passes(mlir)
        assert "module" in optimized
