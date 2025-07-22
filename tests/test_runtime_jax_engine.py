import jax.numpy as jnp
from qmlir.runtime.jax.engine import (
    simulate_from_mlir,
    parse_mlir_operations,
    encode_operations,
    GateID,
    GATE_MATRICES,
    GATE_NAME_TO_ID,
)


class TestGateConstants:
    """Test gate constants and matrices."""

    def test_gate_ids(self):
        """Test that all gate IDs are defined."""
        assert GateID.I == 0
        assert GateID.X == 1
        assert GateID.Y == 2
        assert GateID.Z == 3
        assert GateID.H == 4
        assert GateID.S == 5
        assert GateID.T == 6
        assert GateID.Sdg == 7
        assert GateID.Tdg == 8
        assert GateID.CX == 9
        assert GateID.CY == 10
        assert GateID.CZ == 11
        assert GateID.CCX == 12
        assert GateID.CCY == 13
        assert GateID.CCZ == 14
        assert GateID.RX == 15
        assert GateID.RY == 16
        assert GateID.RZ == 17

    def test_gate_matrices(self):
        """Test that gate matrices are properly defined."""
        # Test I gate matrix
        i_matrix = GATE_MATRICES[GateID.I]
        assert i_matrix.shape == (2, 2)
        assert jnp.allclose(i_matrix, jnp.eye(2, dtype=jnp.complex64))

        # Test X gate matrix
        x_matrix = GATE_MATRICES[GateID.X]
        assert x_matrix.shape == (2, 2)
        assert jnp.allclose(x_matrix, jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64))

        # Test Y gate matrix
        y_matrix = GATE_MATRICES[GateID.Y]
        assert y_matrix.shape == (2, 2)
        expected_y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
        assert jnp.allclose(y_matrix, expected_y)

        # Test Z gate matrix
        z_matrix = GATE_MATRICES[GateID.Z]
        assert z_matrix.shape == (2, 2)
        expected_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
        assert jnp.allclose(z_matrix, expected_z)

        # Test H gate matrix
        h_matrix = GATE_MATRICES[GateID.H]
        assert h_matrix.shape == (2, 2)
        expected_h = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / jnp.sqrt(2)
        assert jnp.allclose(h_matrix, expected_h)

        # Test S gate matrix
        s_matrix = GATE_MATRICES[GateID.S]
        assert s_matrix.shape == (2, 2)
        expected_s = jnp.array([[1, 0], [0, 1j]], dtype=jnp.complex64)
        assert jnp.allclose(s_matrix, expected_s)

        # Test T gate matrix
        t_matrix = GATE_MATRICES[GateID.T]
        assert t_matrix.shape == (2, 2)
        expected_t = jnp.array([[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]], dtype=jnp.complex64)
        assert jnp.allclose(t_matrix, expected_t)

        # Test Sdg gate matrix
        sdg_matrix = GATE_MATRICES[GateID.Sdg]
        assert sdg_matrix.shape == (2, 2)
        expected_sdg = jnp.array([[1, 0], [0, -1j]], dtype=jnp.complex64)
        assert jnp.allclose(sdg_matrix, expected_sdg)

        # Test Tdg gate matrix
        tdg_matrix = GATE_MATRICES[GateID.Tdg]
        assert tdg_matrix.shape == (2, 2)
        expected_tdg = jnp.array([[1, 0], [0, jnp.exp(-1j * jnp.pi / 4)]], dtype=jnp.complex64)
        assert jnp.allclose(tdg_matrix, expected_tdg)

    def test_gate_name_mapping(self):
        """Test gate name to ID mapping."""
        assert GATE_NAME_TO_ID["quantum.i"] == GateID.I
        assert GATE_NAME_TO_ID["quantum.x"] == GateID.X
        assert GATE_NAME_TO_ID["quantum.y"] == GateID.Y
        assert GATE_NAME_TO_ID["quantum.z"] == GateID.Z
        assert GATE_NAME_TO_ID["quantum.h"] == GateID.H
        assert GATE_NAME_TO_ID["quantum.s"] == GateID.S
        assert GATE_NAME_TO_ID["quantum.t"] == GateID.T
        assert GATE_NAME_TO_ID["quantum.sdg"] == GateID.Sdg
        assert GATE_NAME_TO_ID["quantum.tdg"] == GateID.Tdg
        assert GATE_NAME_TO_ID["quantum.cx"] == GateID.CX
        assert GATE_NAME_TO_ID["quantum.cy"] == GateID.CY
        assert GATE_NAME_TO_ID["quantum.cz"] == GateID.CZ
        assert GATE_NAME_TO_ID["quantum.ccx"] == GateID.CCX
        assert GATE_NAME_TO_ID["quantum.ccy"] == GateID.CCY
        assert GATE_NAME_TO_ID["quantum.ccz"] == GateID.CCZ
        assert GATE_NAME_TO_ID["quantum.rx"] == GateID.RX
        assert GATE_NAME_TO_ID["quantum.ry"] == GateID.RY
        assert GATE_NAME_TO_ID["quantum.rz"] == GateID.RZ


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
        assert len(result["final_state"]) == 2
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
        assert len(result["final_state"]) == 4
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
