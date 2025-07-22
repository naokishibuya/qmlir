import jax.numpy as jnp
from qmlir.runtime.jax.circuit import (
    simulate_from_mlir,
    parse_mlir_operations,
    encode_operations,
)
from qmlir.runtime.jax.gate import GateID


class TestMLIRParsing:
    """Test MLIR parsing functions."""

    def test_parse_simple_mlir(self):
        """Test parsing simple MLIR."""
        mlir = """
        module {
          func.func @main() {
            %0 = "quantum.alloc"() : () -> i32
            "quantum.x"(%0) : (i32) -> ()
            return
          }
        }
        """
        operations = parse_mlir_operations(mlir, num_qubits=1, param_values=None, little_endian=True)
        assert len(operations) == 1
        assert operations[0]["gate_name"] == "quantum.x"
        assert operations[0]["qubits"] == [0]

    def test_parse_parametric_mlir(self):
        """Test parsing parametric MLIR."""
        mlir = """
        module {
          func.func @main(%arg0: f64) {
            %0 = "quantum.alloc"() : () -> i32
            "quantum.rx"(%0, %arg0) : (i32, f64) -> ()
            return
          }
        }
        """
        operations = parse_mlir_operations(mlir, num_qubits=1, param_values=[0.5], little_endian=True)
        assert len(operations) == 1
        assert operations[0]["gate_name"] == "quantum.rx"
        assert operations[0]["qubits"] == [0]
        assert operations[0]["params"] == [0.5]

    def test_parse_two_qubit_mlir(self):
        """Test parsing two-qubit MLIR."""
        mlir = """
        module {
          func.func @main() {
            %0 = "quantum.alloc"() : () -> i32
            %1 = "quantum.alloc"() : () -> i32
            "quantum.cx"(%0, %1) : (i32, i32) -> ()
            return
          }
        }
        """
        operations = parse_mlir_operations(mlir, num_qubits=2, param_values=None, little_endian=True)
        assert len(operations) == 1
        assert operations[0]["gate_name"] == "quantum.cx"
        assert operations[0]["qubits"] == [0, 1]

    def test_parse_two_qubit_mlir_with_big_endian(self):
        """Test parsing two-qubit MLIR."""
        mlir = """
        module {
          func.func @main() {
            %0 = "quantum.alloc"() : () -> i32
            %1 = "quantum.alloc"() : () -> i32
            "quantum.cx"(%0, %1) : (i32, i32) -> ()
            return
          }
        }
        """
        operations = parse_mlir_operations(mlir, num_qubits=2, param_values=None, little_endian=False)
        assert len(operations) == 1
        assert operations[0]["gate_name"] == "quantum.cx"
        assert operations[0]["qubits"] == [1, 0]  # Big-endian qubit order

    def test_encode_operations(self):
        """Test operation encoding."""
        operations = [
            {"gate_id": GateID.X, "qubits": [0], "params": []},
            {"gate_id": GateID.H, "qubits": [1], "params": []},
        ]
        encoded = encode_operations(operations)
        assert encoded.shape[0] == 2  # 2 operations
        assert encoded[0, 0] == GateID.X
        assert encoded[0, 1] == 0  # qubit 0
        assert encoded[1, 0] == GateID.H
        assert encoded[1, 1] == 1  # qubit 1


class TestSimulation:
    """Test simulation functions."""

    def test_simulate_empty_circuit(self):
        """Test simulation of empty circuit."""
        mlir = """
        module {
          func.func @main() {
            %0 = "quantum.alloc"() : () -> i32
            return
          }
        }
        """
        result = simulate_from_mlir(mlir, num_qubits=1, param_values=None)
        assert result["num_qubits"] == 1
        assert result["num_operations"] == 0
        assert len(result["statevector"]) == 2
        assert len(result["probabilities"]) == 2

    def test_simulate_x_gate(self):
        """Test simulation of X gate."""
        mlir = """
        module {
          func.func @main() {
            %0 = "quantum.alloc"() : () -> i32
            "quantum.x"(%0) : (i32) -> ()
            return
          }
        }
        """
        result = simulate_from_mlir(mlir, num_qubits=1, param_values=None)
        assert result["num_operations"] == 1
        # X|0⟩ = |1⟩, so probability of |1⟩ should be 1
        assert jnp.allclose(result["probabilities"], jnp.array([0.0, 1.0]))

    def test_simulate_h_gate(self):
        """Test simulation of H gate."""
        mlir = """
        module {
          func.func @main() {
            %0 = "quantum.alloc"() : () -> i32
            "quantum.h"(%0) : (i32) -> ()
            return
          }
        }
        """
        result = simulate_from_mlir(mlir, num_qubits=1, param_values=None)
        assert result["num_operations"] == 1
        # H|0⟩ = (|0⟩ + |1⟩)/√2, so equal probabilities
        expected_probs = jnp.array([0.5, 0.5])
        assert jnp.allclose(result["probabilities"], expected_probs, atol=1e-6)

    def test_simulate_bell_state(self):
        """Test simulation of Bell state circuit."""
        mlir = """
        module {
          func.func @main() {
            %0 = "quantum.alloc"() : () -> i32
            %1 = "quantum.alloc"() : () -> i32
            "quantum.h"(%0) : (i32) -> ()
            "quantum.cx"(%0, %1) : (i32, i32) -> ()
            return
          }
        }
        """
        result = simulate_from_mlir(mlir, num_qubits=2, param_values=None)
        assert result["num_operations"] == 2
        assert len(result["statevector"]) == 4
        assert len(result["probabilities"]) == 4
        # Bell state: (|00⟩ + |11⟩)/√2
        expected_probs = jnp.array([0.5, 0.0, 0.0, 0.5])
        assert jnp.allclose(result["probabilities"], expected_probs, atol=1e-6)

    def test_simulate_parametric_gate(self):
        """Test simulation of parametric gate."""
        mlir = """
        module {
          func.func @main(%arg0: f64) {
            %0 = "quantum.alloc"() : () -> i32
            "quantum.rx"(%0, %arg0) : (i32, f64) -> ()
            return
          }
        }
        """
        result = simulate_from_mlir(mlir, num_qubits=1, param_values=[0.5])
        assert result["num_operations"] == 1
        assert len(result["probabilities"]) == 2
        # RX(0.5)|0⟩ should give specific probabilities
        assert jnp.allclose(result["probabilities"], jnp.array([0.9387913, 0.06120872]), atol=1e-6)


class TestSimulationEdgeCases:
    """Test simulation edge cases."""

    def test_simulate_large_circuit(self):
        """Test simulation of larger circuit."""
        mlir = """
        module {
          func.func @main() {
            %0 = "quantum.alloc"() : () -> i32
            %1 = "quantum.alloc"() : () -> i32
            "quantum.x"(%0) : (i32) -> ()
            "quantum.h"(%1) : (i32) -> ()
            "quantum.cx"(%0, %1) : (i32, i32) -> ()
            return
          }
        }
        """
        result = simulate_from_mlir(mlir, num_qubits=2, param_values=None)
        assert result["num_operations"] == 3
        assert len(result["probabilities"]) == 4

    def test_simulate_without_parameters(self):
        """Test simulation without parameter values."""
        mlir = """
        module {
          func.func @main(%arg0: f64) {
            %0 = "quantum.alloc"() : () -> i32
            "quantum.rx"(%0, %arg0) : (i32, f64) -> ()
            return
          }
        }
        """
        result = simulate_from_mlir(mlir, num_qubits=1, param_values=None)
        assert result["num_operations"] == 1
        # Should use default parameter value (0.0)
        assert jnp.allclose(result["probabilities"], jnp.array([1.0, 0.0]), atol=1e-6)
