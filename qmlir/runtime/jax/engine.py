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
        if gate_name not in GATE_NAME_TO_ID:
            return  # Skip things like "quantum.alloc" - we want to parse only gates here
        gate_id = GATE_NAME_TO_ID[gate_name]

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
    Each row: [gate_id, q0, q1, q2, ..., param0, param1, ...]
    """
    if not operations:
        return jnp.array([])

    max_qubits = max(len(op.get("qubits", [])) for op in operations)
    max_params = max(len(op.get("params", [])) for op in operations)
    operation_size = 1 + max_qubits + max_params  # gate_id + qubits + params

    encoded = []
    for op in operations:
        row = [op["gate_id"]]
        row += op.get("qubits", [])
        row += op.get("params", [])
        row.extend([0.0] * (operation_size - len(row)))  # pad
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

        # Extract gate ID, qubits, and parameters
        gate_id = int(operation[0])
        if gate_id not in GATE_ARITY:
            raise ValueError(f"Unsupported gate ID: {gate_id}")
        n_qubits, n_params = GATE_ARITY[gate_id]

        qubits = operation[1 : 1 + n_qubits].astype(int)
        params = operation[1 + n_qubits : 1 + n_qubits + n_params]

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
