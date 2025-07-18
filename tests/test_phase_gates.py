"""Test the new quantum gates (S, T, Sdg, Tdg, CY, CZ)."""

from qmlir import QuantumCircuit, circuit_to_mlir


class TestPhaseGates:
    """Test the newly added quantum gates."""

    def test_s_gate_creation(self):
        """Test S gate creation and MLIR generation."""
        circuit = QuantumCircuit(1)
        circuit.s(0)

        assert len(circuit.gates) == 1
        assert circuit.gates[0].name == "s"
        assert circuit.gates[0].q == (0,)

        # Test MLIR generation
        mlir_code = circuit_to_mlir(circuit)
        assert '"quantum.s"' in mlir_code

    def test_t_gate_creation(self):
        """Test T gate creation and MLIR generation."""
        circuit = QuantumCircuit(1)
        circuit.t(0)

        assert len(circuit.gates) == 1
        assert circuit.gates[0].name == "t"
        assert circuit.gates[0].q == (0,)

        # Test MLIR generation
        mlir_code = circuit_to_mlir(circuit)
        assert '"quantum.t"' in mlir_code

    def test_sdg_gate_creation(self):
        """Test S-dagger gate creation and MLIR generation."""
        circuit = QuantumCircuit(1)
        circuit.sdg(0)

        assert len(circuit.gates) == 1
        assert circuit.gates[0].name == "sdg"
        assert circuit.gates[0].q == (0,)

        # Test MLIR generation
        mlir_code = circuit_to_mlir(circuit)
        assert '"quantum.sdg"' in mlir_code

    def test_tdg_gate_creation(self):
        """Test T-dagger gate creation and MLIR generation."""
        circuit = QuantumCircuit(1)
        circuit.tdg(0)

        assert len(circuit.gates) == 1
        assert circuit.gates[0].name == "tdg"
        assert circuit.gates[0].q == (0,)

        # Test MLIR generation
        mlir_code = circuit_to_mlir(circuit)
        assert '"quantum.tdg"' in mlir_code

    def test_cy_gate_creation(self):
        """Test controlled-Y gate creation and MLIR generation."""
        circuit = QuantumCircuit(2)
        circuit.cy(0, 1)

        assert len(circuit.gates) == 1
        assert circuit.gates[0].name == "cy"
        assert circuit.gates[0].q == (0, 1)

        # Test MLIR generation
        mlir_code = circuit_to_mlir(circuit)
        assert '"quantum.cy"' in mlir_code

    def test_cz_gate_creation(self):
        """Test controlled-Z gate creation and MLIR generation."""
        circuit = QuantumCircuit(2)
        circuit.cz(0, 1)

        assert len(circuit.gates) == 1
        assert circuit.gates[0].name == "cz"
        assert circuit.gates[0].q == (0, 1)

        # Test MLIR generation
        mlir_code = circuit_to_mlir(circuit)
        assert '"quantum.cz"' in mlir_code

    def test_all_new_gates_together(self):
        """Test all new gates in one circuit."""
        circuit = QuantumCircuit(3)

        # Single-qubit gates
        circuit.s(0)
        circuit.t(0)
        circuit.sdg(1)
        circuit.tdg(1)

        # Multi-qubit gates
        circuit.cy(0, 1)
        circuit.cz(1, 2)

        assert len(circuit.gates) == 6

        # Test MLIR generation
        mlir_code = circuit_to_mlir(circuit)
        assert '"quantum.s"' in mlir_code
        assert '"quantum.t"' in mlir_code
        assert '"quantum.sdg"' in mlir_code
        assert '"quantum.tdg"' in mlir_code
        assert '"quantum.cy"' in mlir_code
        assert '"quantum.cz"' in mlir_code

    def test_method_chaining_with_new_gates(self):
        """Test method chaining works with new gates."""
        circuit = QuantumCircuit(2)

        # Test chaining
        result = circuit.s(0).t(0).sdg(1).tdg(1).cy(0, 1).cz(0, 1)

        # Should return the same circuit object
        assert result is circuit
        assert len(circuit.gates) == 6

    def test_gate_validation_with_new_gates(self):
        """Test qubit validation with new gates."""
        circuit = QuantumCircuit(2)

        # Test invalid qubit indices
        try:
            circuit.s(2)  # Should fail - qubit 2 doesn't exist
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        try:
            circuit.cy(0, 2)  # Should fail - qubit 2 doesn't exist
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        try:
            circuit.cz(0, 0)  # Should fail - same control and target
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
