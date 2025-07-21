"""JAX quantum simulation engine.

This module provides the core JAX-based quantum simulation engine with
gate matrices, operations, and MLIR parsing for the JaxSimulator.
"""

import jax
import jax.numpy as jnp
import re
from enum import IntEnum
from typing import Dict, List, Optional


# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)


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
    "quantum.s": GateID.S_GATE,
    "quantum.t": GateID.T_GATE,
    "quantum.sdg": GateID.S_DAGGER,
    "quantum.tdg": GateID.T_DAGGER,
    "quantum.cx": GateID.CNOT,
    "quantum.cy": GateID.CONTROLLED_Y,
    "quantum.cz": GateID.CONTROLLED_Z,
    "quantum.rx": GateID.ROTATION_X,
    "quantum.ry": GateID.ROTATION_Y,
    "quantum.rz": GateID.ROTATION_Z,
}


def simulate_from_mlir(
    mlir_string: str,
    num_qubits: int,
    param_values: List[float],
    little_endian: bool = True,
) -> Dict:
    """
    Simulate quantum circuit from MLIR string.

    Args:
        mlir_string: MLIR code as string
        num_qubits: Number of qubits in the circuit
        param_values: List of parameter values for parametric gates
        little_endian: Whether to format bitstrings in little-endian order

    Returns:
        Dictionary with simulation results
    """
    # Parse MLIR operations
    operations = parse_mlir_operations(mlir_string, num_qubits, param_values, little_endian)

    # Encode operations for JAX simulation
    encoded_ops = encode_operations(operations)
    print()
    print("Operations:", operations)
    print("Encoded ops:", encoded_ops)
    # Simulate circuit
    final_state = simulate_circuit(encoded_ops, num_qubits, None)

    # Calculate probabilities
    probabilities = jnp.abs(final_state) ** 2

    return {
        "final_state": final_state,
        "probabilities": probabilities,
        "num_qubits": num_qubits,
        "num_operations": len(operations),
    }


def parse_mlir_operations(
    mlir_string: str, num_qubits: int, param_values: List[float], little_endian: bool
) -> List[Dict]:
    """
    Parse MLIR string and extract quantum operations.

    Args:
        mlir_string: MLIR code as string
        num_qubits: Number of qubits in the circuit
        param_values: List of parameter values for parametric gates
        little_endian: Whether to format bitstrings in little-endian order

    Returns:
        List of operation dictionaries with gate info
    """
    operations = []
    lines = mlir_string.strip().split("\n")

    # Track parameter mapping
    param_map = {}

    def parse_args_and_append_op(gate_name, args):
        # Parse arguments
        arg_vars = [arg.strip() for arg in args.split(",")] if args.strip() else []

        # Map qubit variables to consistent indices
        qubit_indices = []
        params = []
        for arg_var in arg_vars:
            if arg_var in param_map:
                # This is a parameter argument
                param_idx = param_map[arg_var]
                if param_values and param_idx < len(param_values):
                    params.append(param_values[param_idx])
                else:
                    params.append(0.0)  # Default value
            else:
                # Parse SSA name like "%1" to integer index 1
                qubit_index = int(arg_var.replace("%", ""))
                if not little_endian:
                    # Reverse qubit index for big-endian
                    qubit_index = num_qubits - 1 - qubit_index
                qubit_indices.append(qubit_index)

        # Get gate ID
        gate_id = GATE_NAME_TO_ID.get(gate_name, GateID.IDENTITY)
        operations.append({"gate_id": gate_id, "qubits": qubit_indices, "params": params, "gate_name": gate_name})

    for line in lines:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("module") or line.startswith("}"):
            continue

        # Match function arguments (parameters)
        # Format: func.func @main(%arg0: f64)
        func_match = re.match(r"func\.func\s+@\w+\(([^)]*)\)", line)
        if func_match:
            args_str = func_match.group(1)
            if args_str.strip():
                args = [arg.strip() for arg in args_str.split(",")]
                for i, arg in enumerate(args):
                    if ":" in arg:
                        param_name = arg.split(":")[0].strip()
                        param_map[param_name] = i
            continue

        # Match quantum operations with return values (old format)
        # Format: %q0 = "quantum.x"(%q0) : (!quantum.qubit) -> !quantum.qubit
        match = re.match(r'%(\w+)\s*=\s*"([^"]+)"\(([^)]*)\)\s*:\s*\(([^)]*)\)\s*->\s*([^)]+)', line)
        if match:
            result_var, gate_name, args, input_types, output_type = match.groups()
            parse_args_and_append_op(gate_name, args)
            continue

        # Match quantum operations without return values (current format)
        # Format: "quantum.h"(%0) : (i32) -> ()
        match = re.match(r'"([^"]+)"\(([^)]*)\)\s*:\s*\(([^)]*)\)\s*->\s*\(\)', line)
        if match:
            gate_name, args, input_types = match.groups()
            parse_args_and_append_op(gate_name, args)

    return operations


def encode_operations(operations: List[Dict]) -> jnp.ndarray:
    """
    Encode operations into JAX array format.

    Args:
        operations: List of operation dictionaries

    Returns:
        JAX array of operations
    """
    if not operations:
        return jnp.array([])

    # Each operation: [gate_id, qubit0, qubit1, param0, param1, ...]
    max_params = max(len(op.get("params", [])) for op in operations)
    operation_size = 3 + max_params  # gate_id + 2 qubits + params

    encoded = []
    for op in operations:
        row = [op["gate_id"]] + op["qubits"] + op.get("params", [])
        # Pad with zeros to match operation_size
        row.extend([0.0] * (operation_size - len(row)))
        encoded.append(row)

    return jnp.array(encoded, dtype=jnp.float64)


def simulate_circuit(
    operations: jnp.ndarray,
    num_qubits: int,
    initial_state: Optional[jnp.ndarray],
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
        params = operation[2:]  # Parameters start from index 2 (after qubits)

        # Apply gate using vectorized operations
        state = apply_gate_vectorized(state, gate_id, qubits, params, num_qubits)

    return state


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
        return _apply_single_gate(state, qubits, num_qubits, gate_matrix)
    elif gate_id == GateID.CNOT:
        return _apply_cx(state, qubits, num_qubits)
    elif gate_id == GateID.CONTROLLED_Y:
        return _apply_cy(state, qubits, num_qubits)
    elif gate_id == GateID.CONTROLLED_Z:
        return _apply_cz(state, qubits, num_qubits)
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

        ctrl_bit = (i >> control) & 1
        if ctrl_bit == 1:
            j = i ^ (1 << target)  # flip target bit

            if i == j:
                v = jnp.array([state[i], 0])
                new_v = gate_matrix @ v
                result = result.at[i].add(new_v[0])
            else:
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
    return _apply_controlled_gate(state, qubits, GATE_MATRICES[GateID.PAULI_X], num_qubits)


def _apply_cy(
    state: jnp.ndarray,
    qubits: jnp.ndarray,
    num_qubits: int,
) -> jnp.ndarray:
    """Apply controlled-Y gate."""
    return _apply_controlled_gate(state, qubits, GATE_MATRICES[GateID.PAULI_Y], num_qubits)


def _apply_cz(
    state: jnp.ndarray,
    qubits: jnp.ndarray,
    num_qubits: int,
) -> jnp.ndarray:
    """Apply controlled-Z gate."""
    return _apply_controlled_gate(state, qubits, GATE_MATRICES[GateID.PAULI_Z], num_qubits)


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
