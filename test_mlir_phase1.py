"""Test MLIR generation for parametric circuits."""

from qmlir import QuantumCircuit, Parameter, circuit_to_mlir


def test_basic_parametric_mlir():
    """Test basic parametric circuit MLIR generation."""
    circuit = QuantumCircuit(2)
    theta = Parameter("theta")
    phi = Parameter("phi")

    # Create a simple parametric circuit
    circuit.rx(0, theta)
    circuit.ry(1, phi)
    circuit.h(0)

    try:
        mlir_code = circuit_to_mlir(circuit, "test_parametric")
        print("✓ Basic parametric MLIR generation succeeded")
        print("\nGenerated MLIR:")
        print(mlir_code)

        # Check that function has parameters
        assert "test_parametric(" in mlir_code
        assert "f64" in mlir_code  # Should have f64 parameters
        return True
    except Exception as e:
        print(f"✗ Basic parametric MLIR generation failed: {e}")
        return False


def test_parameter_reuse_mlir():
    """Test parameter reuse in MLIR generation."""
    circuit = QuantumCircuit(2)
    theta = Parameter("theta")

    # Reuse the same parameter
    circuit.rx(0, theta)
    circuit.rz(1, theta)

    try:
        mlir_code = circuit_to_mlir(circuit, "test_reuse")
        print("✓ Parameter reuse MLIR generation succeeded")
        print("\nGenerated MLIR:")
        print(mlir_code)

        # Should have only one f64 parameter since theta is reused
        # Count occurrences of f64 in function signature
        signature_line = [line for line in mlir_code.split("\n") if "test_reuse" in line][0]
        f64_count = signature_line.count("f64")
        assert f64_count == 1, f"Expected 1 f64 parameter, got {f64_count}"
        return True
    except Exception as e:
        print(f"✗ Parameter reuse MLIR generation failed: {e}")
        return False


def test_no_parameters_mlir():
    """Test that non-parametric circuits still work."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    try:
        mlir_code = circuit_to_mlir(circuit, "test_no_params")
        print("✓ Non-parametric MLIR generation succeeded")
        print("\nGenerated MLIR:")
        print(mlir_code)

        # Should have no parameters
        assert "test_no_params()" in mlir_code
        return True
    except Exception as e:
        print(f"✗ Non-parametric MLIR generation failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Phase 1 MLIR generation...")

    success = True
    success &= test_basic_parametric_mlir()
    print()
    success &= test_parameter_reuse_mlir()
    print()
    success &= test_no_parameters_mlir()

    if success:
        print("\n🎉 All MLIR generation tests passed!")
    else:
        print("\n❌ Some MLIR generation tests failed.")
