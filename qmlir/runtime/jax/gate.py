import jax
import jax.numpy as jnp
from enum import IntEnum


# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)


class GateID(IntEnum):
    """Enumeration of quantum gate identifiers for JAX simulation."""

    # fmt: off
    I   =  0  # Identity gate  # noqa: E741
    X   =  1  # Pauli-X gate (NOT gate)
    Y   =  2  # Pauli-Y gate
    Z   =  3  # Pauli-Z gate
    H   =  4  # Hadamard gate
    S   =  5  # Phase gate
    T   =  6  # T gate (π/8 gate)
    Sdg =  7  # S dagger (conjugate transpose of S)
    Tdg =  8  # T dagger (conjugate transpose of T)
    CX  =  9  # CNOT, controlled-X gate
    CY  = 10  # Controlled-Y gate
    CZ  = 11  # Controlled-Z gate
    CCX = 12  # Toffoli gate
    CCY = 13  # Double controlled Y gate
    CCZ = 14  # Double controlled Z gate
    RX  = 15  # Single-qubit rotation around X axis
    RY  = 16  # Single-qubit rotation around Y axis
    RZ  = 17  # Single-qubit rotation around Z axis
    # fmt: on


# Number of qubits and params for each gate type
# fmt: off
GATE_ARITY = {
    GateID.I  : (1, 0),
    GateID.X  : (1, 0),
    GateID.Y  : (1, 0),
    GateID.Z  : (1, 0),
    GateID.H  : (1, 0),
    GateID.S  : (1, 0),
    GateID.T  : (1, 0),
    GateID.Sdg: (1, 0),
    GateID.Tdg: (1, 0),
    GateID.CX : (2, 0),
    GateID.CY : (2, 0),
    GateID.CZ : (2, 0),
    GateID.CCX: (3, 0),
    GateID.CCY: (3, 0),
    GateID.CCZ: (3, 0),
    GateID.RX : (1, 1),
    GateID.RY : (1, 1),
    GateID.RZ : (1, 1),
}
# fmt: on


# Pre-define all gate matrices as JAX arrays (optimized for JIT)
GATE_MATRICES = {
    GateID.I: jnp.eye(2, dtype=jnp.complex64),
    GateID.X: jnp.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=jnp.complex64,
    ),
    GateID.Y: jnp.array(
        [
            [0, -1j],
            [1j, 0],
        ],
        dtype=jnp.complex64,
    ),
    GateID.Z: jnp.array(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=jnp.complex64,
    ),
    GateID.H: jnp.array(
        [
            [1, 1],
            [1, -1],
        ],
        dtype=jnp.complex64,
    )
    / jnp.sqrt(2),
    GateID.S: jnp.array(
        [
            [1, 0],
            [0, 1j],
        ],
        dtype=jnp.complex64,
    ),
    GateID.T: jnp.array(
        [
            [1, 0],
            [0, jnp.exp(1j * jnp.pi / 4)],
        ],
        dtype=jnp.complex64,
    ),
    GateID.Sdg: jnp.array(
        [
            [1, 0],
            [0, -1j],
        ],
        dtype=jnp.complex64,
    ),
    GateID.Tdg: jnp.array(
        [
            [1, 0],
            [0, jnp.exp(-1j * jnp.pi / 4)],
        ],
        dtype=jnp.complex64,
    ),
}

# Gate name to ID mapping
GATE_NAME_TO_ID = {
    "quantum.i": GateID.I,
    "quantum.x": GateID.X,
    "quantum.y": GateID.Y,
    "quantum.z": GateID.Z,
    "quantum.h": GateID.H,
    "quantum.s": GateID.S,
    "quantum.t": GateID.T,
    "quantum.sdg": GateID.Sdg,
    "quantum.tdg": GateID.Tdg,
    "quantum.cx": GateID.CX,
    "quantum.cy": GateID.CY,
    "quantum.cz": GateID.CZ,
    "quantum.ccx": GateID.CCX,
    "quantum.ccy": GateID.CCY,
    "quantum.ccz": GateID.CCZ,
    "quantum.rx": GateID.RX,
    "quantum.ry": GateID.RY,
    "quantum.rz": GateID.RZ,
}


def apply_gate_vectorized(
    state: jnp.ndarray,
    gate_id: int,
    qubits: jnp.ndarray,
    params: jnp.ndarray,
    num_qubits: int,
) -> jnp.ndarray:
    """
    Apply quantum gate using JAX vectorized operations.

    Args:
        state: Current quantum state vector
        gate_id: Gate identifier (0-17)
        qubits: Qubit indices to apply the gate to
        params: Gate parameters for rotation gates
        num_qubits: Total number of qubits

    Returns:
        Updated quantum state vector
    """
    # Simple implementation for basic gates
    if gate_id <= GateID.Tdg:  # Single-qubit gates
        gate_matrix = GATE_MATRICES.get(gate_id, GATE_MATRICES[GateID.I])
        return _apply_single_gate(state, qubits, num_qubits, gate_matrix)
    elif gate_id == GateID.CX:
        return _apply_cx(state, qubits, num_qubits)
    elif gate_id == GateID.CY:
        return _apply_cy(state, qubits, num_qubits)
    elif gate_id == GateID.CZ:
        return _apply_cz(state, qubits, num_qubits)
    elif gate_id == GateID.CCX:
        return _apply_ccx(state, qubits, num_qubits)
    elif gate_id == GateID.CCY:
        return _apply_ccy(state, qubits, num_qubits)
    elif gate_id == GateID.CCZ:
        return _apply_ccz(state, qubits, num_qubits)
    elif gate_id == GateID.RX:
        return _apply_rx(state, qubits, params, num_qubits)
    elif gate_id == GateID.RY:
        return _apply_ry(state, qubits, params, num_qubits)
    elif gate_id == GateID.RZ:
        return _apply_rz(state, qubits, params, num_qubits)
    else:
        # For other gates, return state unchanged for now
        return state


def _apply_single_gate(
    state: jnp.ndarray,
    qubits: jnp.ndarray,
    num_qubits: int,
    gate_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Apply single-qubit gate using tensor product operations."""
    target_qubit = int(qubits[0])  # target qubit index

    full_op = None
    for i in range(num_qubits):
        # If this is the target qubit, apply the gate matrix
        if i == target_qubit:
            op = gate_matrix
        else:
            op = jnp.eye(2, dtype=jnp.complex64)  # Identity for other qubits

        # If this is the first operation, set it directly
        if full_op is None:
            full_op = op
        else:
            # Tensor product with previous operations
            full_op = jnp.kron(op, full_op)

    return full_op @ state


def _apply_controlled_gate(
    state: jnp.ndarray,
    qubits: jnp.ndarray,
    gate_matrix: jnp.ndarray,
    num_qubits: int,
) -> jnp.ndarray:
    # Control and target qubits
    control, target = int(qubits[0]), int(qubits[1])
    if control == target:
        raise ValueError("Control and target qubits must be different.")

    # Create the controlled gate matrix
    dim = 1 << num_qubits
    result = jnp.zeros_like(state)
    visited = set()

    for i in range(dim):
        if i in visited:
            continue

        if ((i >> control) & 1) == 1:
            j = i ^ (1 << target)  # Flip target bit

            v = jnp.array([state[i], state[j]])
            new_v = gate_matrix @ v

            result = result.at[i].add(new_v[0])
            result = result.at[j].add(new_v[1])

            visited.add(i)
            visited.add(j)
        else:
            result = result.at[i].add(state[i])

    return result


def _apply_cx(
    state: jnp.ndarray,
    qubits: jnp.ndarray,
    num_qubits: int,
) -> jnp.ndarray:
    """Apply CNOT gate."""
    return _apply_controlled_gate(state, qubits, GATE_MATRICES[GateID.X], num_qubits)


def _apply_cy(
    state: jnp.ndarray,
    qubits: jnp.ndarray,
    num_qubits: int,
) -> jnp.ndarray:
    """Apply controlled-Y gate."""
    return _apply_controlled_gate(state, qubits, GATE_MATRICES[GateID.Y], num_qubits)


def _apply_cz(
    state: jnp.ndarray,
    qubits: jnp.ndarray,
    num_qubits: int,
) -> jnp.ndarray:
    """Apply controlled-Z gate."""
    return _apply_controlled_gate(state, qubits, GATE_MATRICES[GateID.Z], num_qubits)


def _apply_controlled_controlled_gate(
    state: jnp.ndarray,
    qubits: jnp.ndarray,
    gate_matrix: jnp.ndarray,
    num_qubits: int,
) -> jnp.ndarray:
    """Apply controlled-controlled gate (Toffoli or similar)."""
    control1, control2, target = int(qubits[0]), int(qubits[1]), int(qubits[2])
    if control1 == control2 or control1 == target or control2 == target:
        raise ValueError("All qubits must be distinct.")

    # Create the controlled-controlled gate matrix
    dim = 1 << num_qubits
    result = jnp.zeros_like(state)
    visited = set()

    for i in range(dim):
        if i in visited:
            continue

        ctrl1_bit = (i >> control1) & 1
        ctrl2_bit = (i >> control2) & 1

        if ctrl1_bit == 1 and ctrl2_bit == 1:
            j = i ^ (1 << target)  # flip target bit

            v = jnp.array([state[i], state[j]])
            new_v = gate_matrix @ v

            result = result.at[i].add(new_v[0])
            result = result.at[j].add(new_v[1])

            visited.add(i)
            visited.add(j)
        else:
            result = result.at[i].add(state[i])

    return result


def _apply_ccx(
    state: jnp.ndarray,
    qubits: jnp.ndarray,
    num_qubits: int,
) -> jnp.ndarray:
    """Apply Toffoli gate (CCX)."""
    return _apply_controlled_controlled_gate(state, qubits, GATE_MATRICES[GateID.X], num_qubits)


def _apply_ccy(
    state: jnp.ndarray,
    qubits: jnp.ndarray,
    num_qubits: int,
) -> jnp.ndarray:
    """Apply double controlled Y gate (CCY)."""
    return _apply_controlled_controlled_gate(state, qubits, GATE_MATRICES[GateID.Y], num_qubits)


def _apply_ccz(
    state: jnp.ndarray,
    qubits: jnp.ndarray,
    num_qubits: int,
) -> jnp.ndarray:
    """Apply double controlled Z gate (CCZ)."""
    return _apply_controlled_controlled_gate(state, qubits, GATE_MATRICES[GateID.Z], num_qubits)


def _apply_rx(
    state: jnp.ndarray,
    qubits: jnp.ndarray,
    params: jnp.ndarray,
    num_qubits: int,
) -> jnp.ndarray:
    """Apply RX rotation gate."""
    theta = params[0]
    rx_matrix = jnp.array(
        [[jnp.cos(theta / 2), -1j * jnp.sin(theta / 2)], [-1j * jnp.sin(theta / 2), jnp.cos(theta / 2)]],
        dtype=jnp.complex64,
    )
    return _apply_single_gate(state, qubits, num_qubits, rx_matrix)


def _apply_ry(
    state: jnp.ndarray,
    qubits: jnp.ndarray,
    params: jnp.ndarray,
    num_qubits: int,
) -> jnp.ndarray:
    """Apply RY rotation gate."""
    theta = params[0]
    ry_matrix = jnp.array(
        [[jnp.cos(theta / 2), -jnp.sin(theta / 2)], [jnp.sin(theta / 2), jnp.cos(theta / 2)]], dtype=jnp.complex64
    )
    return _apply_single_gate(state, qubits, num_qubits, ry_matrix)


def _apply_rz(
    state: jnp.ndarray,
    qubits: jnp.ndarray,
    params: jnp.ndarray,
    num_qubits: int,
) -> jnp.ndarray:
    """Apply RZ rotation gate."""
    theta = params[0]
    rz_matrix = jnp.array([[jnp.exp(-1j * theta / 2), 0], [0, jnp.exp(1j * theta / 2)]], dtype=jnp.complex64)
    return _apply_single_gate(state, qubits, num_qubits, rz_matrix)
