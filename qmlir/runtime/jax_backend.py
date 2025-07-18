"""JAX-based quantum circuit simulator backend.

This module implements high-performance quantum circuit simulation using JAX
for JIT compilation, automatic differentiation, and GPU/TPU acceleration.
"""

import jax.numpy as jnp
import re
from enum import IntEnum
from typing import Dict, List, Optional


class GateID(IntEnum):
    """Enumeration of quantum gate identifiers for JAX simulation."""

    IDENTITY = 0
    PAULI_X = 1
    PAULI_Y = 2
    PAULI_Z = 3
    HADAMARD = 4
    S_GATE = 5
    T_GATE = 6
    S_DAGGER = 7
    T_DAGGER = 8
    CNOT = 9
    CONTROLLED_Y = 10
    CONTROLLED_Z = 11
    ROTATION_X = 12
    ROTATION_Y = 13
    ROTATION_Z = 14


# Pre-define all gate matrices as JAX arrays (optimized for JIT)
GATE_MATRICES = {
    GateID.IDENTITY: jnp.eye(2, dtype=jnp.complex64),
    GateID.PAULI_X: jnp.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=jnp.complex64,
    ),
    GateID.PAULI_Y: jnp.array(
        [
            [0, -1j],
            [1j, 0],
        ],
        dtype=jnp.complex64,
    ),
    GateID.PAULI_Z: jnp.array(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=jnp.complex64,
    ),
    GateID.HADAMARD: jnp.array(
        [
            [1, 1],
            [1, -1],
        ],
        dtype=jnp.complex64,
    )
    / jnp.sqrt(2),
    GateID.S_GATE: jnp.array(
        [
            [1, 0],
            [0, 1j],
        ],
        dtype=jnp.complex64,
    ),
    GateID.T_GATE: jnp.array(
        [
            [1, 0],
            [0, jnp.exp(1j * jnp.pi / 4)],
        ],
        dtype=jnp.complex64,
    ),
    GateID.S_DAGGER: jnp.array(
        [
            [1, 0],
            [0, -1j],
        ],
        dtype=jnp.complex64,
    ),
    GateID.T_DAGGER: jnp.array(
        [
            [1, 0],
            [0, jnp.exp(-1j * jnp.pi / 4)],
        ],
        dtype=jnp.complex64,
    ),
}

# Gate name to ID mapping
GATE_NAME_TO_ID = {
    "quantum.i": GateID.IDENTITY,
    "quantum.x": GateID.PAULI_X,
    "quantum.y": GateID.PAULI_Y,
    "quantum.z": GateID.PAULI_Z,
    "quantum.h": GateID.HADAMARD,
    "s": GateID.S_GATE,
    "t": GateID.T_GATE,
    "sdg": GateID.S_DAGGER,
    "tdg": GateID.T_DAGGER,
    "quantum.cx": GateID.CNOT,
    "cy": GateID.CONTROLLED_Y,
    "cz": GateID.CONTROLLED_Z,
    "rx": GateID.ROTATION_X,
    "ry": GateID.ROTATION_Y,
    "rz": GateID.ROTATION_Z,
}


def simulate_circuit(
    operations: jnp.ndarray, num_qubits: int, initial_state: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Simulate quantum circuit using JAX (non-JIT version for dynamic shapes).

    Args:
        operations: Array of [gate_id, qubit_indices, parameters] for each operation
        num_qubits: Number of qubits in the circuit
        initial_state: Optional initial quantum state vector

    Returns:
        Final quantum state vector
    """
    # Initialize state vector |00...0⟩
    if initial_state is None:
        initial_state = jnp.zeros(2**num_qubits, dtype=jnp.complex64).at[0].set(1.0)

    # Apply operations sequentially (no JIT for dynamic shapes)
    state = initial_state
    for i in range(operations.shape[0]):
        operation = operations[i]
        gate_id = int(operation[0])
        qubits = operation[1:3].astype(int)  # Max 2 qubits for current gates
        params = operation[3:]  # Parameters for rotation gates

        # Apply gate using vectorized operations
        state = apply_gate_vectorized(state, gate_id, qubits, params, num_qubits)

    return state


def apply_gate_vectorized(
    state: jnp.ndarray, gate_id: int, qubits: jnp.ndarray, params: jnp.ndarray, num_qubits: int
) -> jnp.ndarray:
    """
    Apply quantum gate using JAX vectorized operations.

    Args:
        state: Current quantum state vector
        gate_id: Gate identifier (0-14)
        qubits: Qubit indices [qubit0, qubit1] (padded with -1 for single-qubit gates)
        params: Gate parameters for rotation gates
        num_qubits: Total number of qubits

    Returns:
        Updated quantum state vector
    """
    # Simple implementation for basic gates
    if gate_id <= GateID.T_DAGGER:  # Single-qubit gates
        gate_matrix = GATE_MATRICES.get(gate_id, GATE_MATRICES[GateID.IDENTITY])
        return _apply_single_gate(state, qubits, params, num_qubits, gate_matrix)
    elif gate_id == GateID.CNOT:
        return _apply_cnot(state, qubits, params, num_qubits)
    elif gate_id == GateID.ROTATION_X:
        return _apply_rx(state, qubits, params, num_qubits)
    elif gate_id == GateID.ROTATION_Y:
        return _apply_ry(state, qubits, params, num_qubits)
    elif gate_id == GateID.ROTATION_Z:
        return _apply_rz(state, qubits, params, num_qubits)
    else:
        # For other gates, return state unchanged for now
        return state


def _apply_single_gate(
    state: jnp.ndarray, qubits: jnp.ndarray, params: jnp.ndarray, num_qubits: int, gate_matrix: jnp.ndarray
) -> jnp.ndarray:
    """Apply single-qubit gate using tensor product operations."""
    qubit = qubits[0]

    # Build full gate matrix via tensor products
    full_gate = jnp.array([[1.0]], dtype=jnp.complex64)

    for i in range(num_qubits):
        if i == qubit:
            full_gate = jnp.kron(full_gate, gate_matrix)
        else:
            full_gate = jnp.kron(full_gate, jnp.eye(2, dtype=jnp.complex64))

    return full_gate @ state


def _apply_cnot(state: jnp.ndarray, qubits: jnp.ndarray, params: jnp.ndarray, num_qubits: int) -> jnp.ndarray:
    """Apply CNOT gate using JAX-compatible operations."""
    control, target = int(qubits[0]), int(qubits[1])

    # For 2-qubit systems, use simple CNOT matrix
    if num_qubits == 2:
        if control == 0 and target == 1:
            cnot_matrix = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=jnp.complex64)
        else:  # control == 1 and target == 0
            cnot_matrix = jnp.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=jnp.complex64)

        return cnot_matrix @ state
    else:
        # For larger systems, use tensor product approach (simplified)
        # This is a placeholder - would need proper implementation
        return state


def _apply_cy(state: jnp.ndarray, qubits: jnp.ndarray, params: jnp.ndarray, num_qubits: int) -> jnp.ndarray:
    """Apply controlled-Y gate."""
    # Placeholder - similar to CNOT but with Y gate
    return state


def _apply_cz(state: jnp.ndarray, qubits: jnp.ndarray, params: jnp.ndarray, num_qubits: int) -> jnp.ndarray:
    """Apply controlled-Z gate."""
    # Placeholder - similar to CNOT but with Z gate
    return state


def _apply_rx(state: jnp.ndarray, qubits: jnp.ndarray, params: jnp.ndarray, num_qubits: int) -> jnp.ndarray:
    """Apply rotation-X gate with parameter."""
    angle = params[0]

    # RX(θ) = cos(θ/2)I - i*sin(θ/2)X
    cos_half = jnp.cos(angle / 2)
    sin_half = jnp.sin(angle / 2)

    rx_matrix = jnp.array([[cos_half, -1j * sin_half], [-1j * sin_half, cos_half]], dtype=jnp.complex64)

    return _apply_single_gate(state, qubits, params, num_qubits, rx_matrix)


def _apply_ry(state: jnp.ndarray, qubits: jnp.ndarray, params: jnp.ndarray, num_qubits: int) -> jnp.ndarray:
    """Apply rotation-Y gate with parameter."""
    angle = params[0]

    # RY(θ) = cos(θ/2)I - i*sin(θ/2)Y
    cos_half = jnp.cos(angle / 2)
    sin_half = jnp.sin(angle / 2)

    ry_matrix = jnp.array([[cos_half, -sin_half], [sin_half, cos_half]], dtype=jnp.complex64)

    return _apply_single_gate(state, qubits, params, num_qubits, ry_matrix)


def _apply_rz(state: jnp.ndarray, qubits: jnp.ndarray, params: jnp.ndarray, num_qubits: int) -> jnp.ndarray:
    """Apply rotation-Z gate with parameter."""
    angle = params[0]

    # RZ(θ) = exp(-iθ/2)|0⟩⟨0| + exp(iθ/2)|1⟩⟨1|
    exp_neg = jnp.exp(-1j * angle / 2)
    exp_pos = jnp.exp(1j * angle / 2)

    rz_matrix = jnp.array([[exp_neg, 0], [0, exp_pos]], dtype=jnp.complex64)

    return _apply_single_gate(state, qubits, params, num_qubits, rz_matrix)


def parse_mlir_operations(mlir_string: str) -> List[Dict]:
    """
    Parse MLIR string to extract quantum operations.

    Args:
        mlir_string: MLIR representation of quantum circuit

    Returns:
        List of operation dictionaries
    """
    operations = []

    # Simple regex-based parsing (can be improved)
    operation_pattern = r'"(quantum\.\w+)"\(([^)]*)\)'

    for match in re.finditer(operation_pattern, mlir_string):
        op_name = match.group(1)
        operands = match.group(2)

        # Extract qubit indices (simplified)
        qubits = []
        if operands:
            # This is a simplified parser - needs improvement for real MLIR
            qubit_matches = re.findall(r"%(\d+)", operands)
            qubits = [int(q) for q in qubit_matches]

        operations.append(
            {
                "type": op_name,
                "qubits": qubits,
                "parameters": [],  # TODO: Parse parameters
            }
        )

    return operations


def encode_operations(operations: List[Dict]) -> jnp.ndarray:
    """
    Convert parsed operations to JAX-compatible array format.

    Args:
        operations: List of operation dictionaries

    Returns:
        JAX array of shape [num_ops, 6] with [gate_id, qubit0, qubit1, param0, param1, param2]
    """
    encoded = []

    for op in operations:
        gate_id = GATE_NAME_TO_ID.get(op["type"], GateID.IDENTITY)  # Default to identity

        # Pad qubits to 2 elements
        qubits = op["qubits"] + [-1] * (2 - len(op["qubits"]))

        # Pad parameters to 3 elements
        params = op.get("parameters", [])
        if len(params) > 0:
            param_values = [p.value if hasattr(p, "value") else float(p) for p in params]
        else:
            param_values = []
        param_values += [0.0] * (3 - len(param_values))

        encoded.append([gate_id] + qubits + param_values)

    return jnp.array(encoded, dtype=jnp.float32)


def simulate_from_mlir(mlir_string: str, num_qubits: int) -> Dict:
    """
    High-level function to simulate quantum circuit from MLIR string.

    Args:
        mlir_string: MLIR representation of quantum circuit
        num_qubits: Number of qubits in the circuit

    Returns:
        Dictionary with simulation results
    """
    # Parse MLIR operations
    operations = parse_mlir_operations(mlir_string)

    # Encode for JAX
    encoded_ops = encode_operations(operations)

    # Simulate
    final_state = simulate_circuit(encoded_ops, num_qubits)

    return {
        "final_state": final_state,
        "probabilities": jnp.abs(final_state) ** 2,
        "num_qubits": num_qubits,
        "num_operations": len(operations),
    }
