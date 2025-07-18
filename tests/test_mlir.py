"""Test MLIR code generation functionality."""

from qmlir import QuantumCircuit, Parameter
from qmlir.mlir import circuit_to_mlir


class TestMLIRGeneration:
    """Test MLIR code generation."""

    def test_bell_state_mlir(self):
        """Test generating MLIR for Bell state circuit."""
        circuit = QuantumCircuit(2)
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
        circuit = QuantumCircuit(1)
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
        circuit = QuantumCircuit(3)
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
        circuit = QuantumCircuit(1)

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
        circuit = QuantumCircuit(1)
        circuit.h(0)

        mlir_code = circuit_to_mlir(circuit, "my_custom_function")

        assert "func.func @my_custom_function()" in mlir_code


class TestMLIRValidation:
    """Test that generated MLIR is valid."""

    def test_mlir_structure(self):
        """Test that generated MLIR has correct structure."""
        circuit = QuantumCircuit(2)
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
        circuit = QuantumCircuit(3)
        circuit.h(0).cx(0, 1).x(2)  # Uses qubits 0, 1, 2

        mlir_code = circuit_to_mlir(circuit, "test_allocation")

        # Should allocate 3 qubits
        assert mlir_code.count('"quantum.alloc"()') == 3

        # Should have correct SSA value pattern
        assert "%0 = " in mlir_code
        assert "%1 = " in mlir_code
        assert "%2 = " in mlir_code


class TestParametricMLIRGeneration:
    """Test MLIR code generation for parametric circuits."""

    def test_basic_parametric_mlir(self):
        """Test basic parametric circuit MLIR generation."""
        circuit = QuantumCircuit(2)
        theta = Parameter(1.57, "theta")
        phi = Parameter(0.785, "phi")

        # Create a simple parametric circuit
        circuit.rx(0, theta)
        circuit.ry(1, phi)
        circuit.h(0)

        mlir_code = circuit_to_mlir(circuit, "test_parametric")

        # Check function has parameters
        assert "test_parametric(" in mlir_code
        assert "f64" in mlir_code  # Should have f64 parameters
        assert "%arg0: f64" in mlir_code
        assert "%arg1: f64" in mlir_code

        # Check parametric gates are generated
        assert '"quantum.rx"(' in mlir_code
        assert '"quantum.ry"(' in mlir_code
        assert '"quantum.h"(' in mlir_code

        # Check parameter usage
        assert "%arg0" in mlir_code
        assert "%arg1" in mlir_code

    def test_parameter_reuse_mlir(self):
        """Test parameter reuse in MLIR generation."""
        circuit = QuantumCircuit(2)
        theta = Parameter(1.57, "theta")

        # Reuse the same parameter
        circuit.rx(0, theta)
        circuit.rz(1, theta)

        mlir_code = circuit_to_mlir(circuit, "test_reuse")

        # Should have only one f64 parameter since theta is reused
        signature_line = [line for line in mlir_code.split("\n") if "test_reuse" in line and "func.func" in line][0]
        f64_count = signature_line.count("f64")
        assert f64_count == 1, f"Expected 1 f64 parameter, got {f64_count}"

        # Both gates should use the same argument
        assert '"quantum.rx"(' in mlir_code
        assert '"quantum.rz"(' in mlir_code
        assert mlir_code.count("%arg0") >= 2  # Used in both gates

    def test_no_parameters_mlir(self):
        """Test that non-parametric circuits still work."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        mlir_code = circuit_to_mlir(circuit, "test_no_params")

        # Should have no parameters
        assert "test_no_params()" in mlir_code
        assert "f64" not in mlir_code  # No f64 parameters
        assert "%arg" not in mlir_code  # No arguments

        # Should still generate gates correctly
        assert '"quantum.h"(' in mlir_code
        assert '"quantum.cx"(' in mlir_code

    def test_mixed_parametric_and_non_parametric_mlir(self):
        """Test mixed parametric and non-parametric gates."""
        circuit = QuantumCircuit(3)
        theta = Parameter(1.57, "theta")

        # Mix of parametric and non-parametric gates
        circuit.h(0)
        circuit.rx(1, theta)
        circuit.cx(0, 1)
        circuit.ry(2, theta)  # Reuse parameter
        circuit.z(2)

        mlir_code = circuit_to_mlir(circuit, "test_mixed")

        # Should have one parameter (theta)
        assert "test_mixed(%arg0: f64)" in mlir_code

        # Should have all gate types
        assert '"quantum.h"(' in mlir_code
        assert '"quantum.rx"(' in mlir_code
        assert '"quantum.cx"(' in mlir_code
        assert '"quantum.ry"(' in mlir_code
        assert '"quantum.z"(' in mlir_code

        # Parameter should be used in rx and ry gates
        rx_and_ry_count = mlir_code.count("%arg0")
        assert rx_and_ry_count >= 2  # Used in both rx and ry gates

    def test_multiple_unique_parameters_mlir(self):
        """Test multiple unique parameters generate correct function signature."""
        circuit = QuantumCircuit(4)
        theta = Parameter(1.57, "theta")
        phi = Parameter(0.785, "phi")
        omega = Parameter(3.14, "omega")

        # Use all three parameters
        circuit.rx(0, theta)
        circuit.ry(1, phi)
        circuit.rz(2, omega)
        circuit.rx(3, theta)  # Reuse theta

        mlir_code = circuit_to_mlir(circuit, "test_multi_params")

        # Should have three parameters
        assert "test_multi_params(%arg0: f64, %arg1: f64, %arg2: f64)" in mlir_code

        # Should use all three arguments
        assert "%arg0" in mlir_code  # theta
        assert "%arg1" in mlir_code  # phi
        assert "%arg2" in mlir_code  # omega

        # Should have correct gate types
        assert '"quantum.rx"(' in mlir_code
        assert '"quantum.ry"(' in mlir_code
        assert '"quantum.rz"(' in mlir_code
