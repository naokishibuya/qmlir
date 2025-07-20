"""MLIR Transpilation and Optimization for Quantum Circuits

This module handles the transpilation of quantum circuit representation to MLIR
representation using the quantum dialect, and provides optimization capabilities.
"""

import subprocess
from ..circuit import QuantumCircuit
from ..operator import Operator
from .config import get_quantum_opt_path  # This will also ensure LLVM/MLIR bindings
from mlir import ir


def circuit_to_mlir(circuit: QuantumCircuit, function_name: str = "main") -> str:
    """Convert a quantum circuit to MLIR representation.

    Args:
        circuit: The quantum circuit to convert
        function_name: Name of the generated function

    Returns:
        MLIR string representation of the circuit
    """
    param_index_map = {}
    for operator in circuit.operators:
        if not isinstance(operator, Operator):
            raise AssertionError(f"Unknown operator: {operator.__class__.__name__}")
        for param in operator.parameters:
            if param.id not in param_index_map:
                param_index_map[param.id] = len(param_index_map)

    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        ctx.enable_multithreading(False)

        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            f64 = ir.F64Type.get()
            func_type = ir.FunctionType.get([f64] * len(param_index_map), [])
            func_op = ir.Operation.create(
                "func.func",
                attributes={
                    "sym_name": ir.StringAttr.get(function_name),
                    "function_type": ir.TypeAttr.get(func_type),
                },
                regions=1,
            )

            block = func_op.regions[0].blocks.append()
            for _ in range(len(param_index_map)):
                block.add_argument(f64, ir.Location.unknown())

            with ir.InsertionPoint(block):
                i32 = ir.IntegerType.get_signless(32)
                ssa_qubits = {
                    i: ir.Operation.create("quantum.alloc", results=[i32]).result for i in range(circuit.num_qubits)
                }

                for operator in circuit.operators:
                    name = operator.name.lower()
                    qubits = [ssa_qubits[q] for q in operator.qubits]
                    operands = qubits.copy()

                    if operator.parameters:
                        for p in operator.parameters:
                            idx = param_index_map[p.id]
                            operands.append(block.arguments[idx])

                    ir.Operation.create(f"quantum.{name}", operands=operands)

                ir.Operation.create("func.return")

        return str(module)


def apply_passes(mlir_code, *args, timeout=10):
    """Apply MLIR optimization passes using quantum-opt.

    Args:
        mlir_code (str): The MLIR code to process.
        *args: Additional command-line arguments to pass to quantum-opt.
        timeout (int): Timeout in seconds for the subprocess call.

    Returns:
        str: The optimized MLIR code.

    Raises:
        RuntimeError: If quantum-opt is not found or optimization fails.
    """
    quantum_opt_path = get_quantum_opt_path()
    if not args:
        args = ["--quantum-cancel-self-inverse"]

    command = [quantum_opt_path] + list(args)
    result = subprocess.run(command, input=mlir_code, capture_output=True, text=True, timeout=timeout)

    if result.returncode != 0:
        raise RuntimeError(f"quantum-opt failed: {result.stderr}")

    return result.stdout
