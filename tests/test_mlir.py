"""Test MLIR code generation functionality."""

from qmlir import Circuit, circuit_to_mlir


class TestMLIRGeneration:
    """Test MLIR code generation."""

    def test_bell_state_mlir(self):
        """Test generating MLIR for Bell state circuit."""
        circuit = Circuit()
        circuit.h(0).cx(0, 1)

        mlir_code = circuit_to_mlir(circuit, "test_bell_state")

        # Check that the MLIR contains expected elements
        assert "module {" in mlir_code
        assert "func.func @test_bell_state()" in mlir_code
        assert '"quantum.alloc"()' in mlir_code
        assert '"quantum.h"(' in mlir_code
        assert '"quantum.cx"(' in mlir_code
        assert "return" in mlir_code
        assert "}" in mlir_code

    def test_double_x_mlir(self):
        """Test generating MLIR for double X circuit."""
        circuit = Circuit()
        circuit.x(0).x(0)

        mlir_code = circuit_to_mlir(circuit, "test_double_x")

        # Check that the MLIR contains expected elements
        assert "module {" in mlir_code
        assert "func.func @test_double_x()" in mlir_code
        assert '"quantum.alloc"()' in mlir_code
        assert '"quantum.x"(' in mlir_code
        assert "return" in mlir_code

        # Should have two X gates
        assert mlir_code.count('"quantum.x"(') == 2

    def test_custom_circuit_mlir(self):
        """Test generating MLIR for a custom multi-qubit circuit."""
        circuit = Circuit()
        circuit.h(0).h(1).x(2).cx(0, 1).cx(1, 2)

        mlir_code = circuit_to_mlir(circuit, "test_custom")

        # Check that the MLIR contains expected elements
        assert "module {" in mlir_code
        assert "func.func @test_custom()" in mlir_code
        assert '"quantum.alloc"()' in mlir_code
        assert '"quantum.h"(' in mlir_code
        assert '"quantum.x"(' in mlir_code
        assert '"quantum.cx"(' in mlir_code
        assert "return" in mlir_code

        # Should have correct number of operations
        assert mlir_code.count('"quantum.alloc"()') == 3  # 3 qubits
        assert mlir_code.count('"quantum.h"(') == 2  # 2 H gates
        assert mlir_code.count('"quantum.x"(') == 1  # 1 X gate
        assert mlir_code.count('"quantum.cx"(') == 2  # 2 CX gates

    def test_empty_circuit_mlir(self):
        """Test generating MLIR for an empty circuit."""
        circuit = Circuit()

        mlir_code = circuit_to_mlir(circuit, "test_empty")

        # Even empty circuit should have basic structure
        assert "module {" in mlir_code
        assert "func.func @test_empty()" in mlir_code
        assert "return" in mlir_code
        assert "}" in mlir_code

        # Should not have any quantum operations
        assert '"quantum.alloc"()' not in mlir_code
        assert '"quantum.h"(' not in mlir_code
        assert '"quantum.x"(' not in mlir_code
        assert '"quantum.cx"(' not in mlir_code

    def test_function_name_parameter(self):
        """Test that function name parameter works correctly."""
        circuit = Circuit()
        circuit.h(0)

        mlir_code = circuit_to_mlir(circuit, "my_custom_function")

        assert "func.func @my_custom_function()" in mlir_code


class TestMLIRValidation:
    """Test that generated MLIR is valid."""

    def test_mlir_structure(self):
        """Test that generated MLIR has correct structure."""
        circuit = Circuit()
        circuit.h(0).cx(0, 1)

        mlir_code = circuit_to_mlir(circuit, "test_structure")

        # Should start with module
        assert mlir_code.strip().startswith("module {")

        # Should end with closing brace
        assert mlir_code.strip().endswith("}")

        # Should have matching braces
        open_braces = mlir_code.count("{")
        close_braces = mlir_code.count("}")
        assert open_braces == close_braces

    def test_qubit_allocation(self):
        """Test that qubits are allocated correctly."""
        circuit = Circuit()
        circuit.h(0).cx(0, 1).x(2)  # Uses qubits 0, 1, 2

        mlir_code = circuit_to_mlir(circuit, "test_allocation")

        # Should allocate 3 qubits
        assert mlir_code.count('"quantum.alloc"()') == 3

        # Should have correct SSA value pattern
        assert "%0 = " in mlir_code
        assert "%1 = " in mlir_code
        assert "%2 = " in mlir_code
