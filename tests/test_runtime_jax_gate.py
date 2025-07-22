import jax.numpy as jnp
from qmlir.runtime.jax.gate import (
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
