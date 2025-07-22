import jax.numpy as jnp
import re
from typing import Dict, List, Optional
from ...circuit import QuantumCircuit
from ...mlir import circuit_to_mlir, apply_passes
from .gate import GATE_NAME_TO_ID, GATE_ARITY, apply_gate_vectorized


def simulate_circuit(circuit: QuantumCircuit, optimize_circuit: bool) -> Dict:
    """Internal method to simulate a circuit and return results."""
    # Step 1: Transpile circuit to MLIR
    mlir_code = circuit_to_mlir(circuit)

    # Step 2: Apply optimization passes if requested
    if optimize_circuit:
        mlir_code = apply_passes(mlir_code)
    circuit.compiled_mlir = mlir_code.strip()  # Store compiled MLIR in circuit for reference

    # Step 3: Collect parameter values from circuit
    param_values = []
    param_ids_seen = set()
    for gate in circuit.gates:
        for param in gate.parameters:
            if param.id not in param_ids_seen:
                param_values.append(param.value)
                param_ids_seen.add(param.id)

    # Step 4: Simulate with JAX runtime
    return simulate_from_mlir(mlir_code, circuit.num_qubits, param_values, circuit.little_endian)


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
    statevector = simulate_statevector(encoded_ops, num_qubits, None)

    # Calculate probabilities
    probabilities = jnp.abs(statevector) ** 2

    return {
        "statevector": statevector,
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


def simulate_statevector(
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
