"""Code generation for quantum circuits to MLIR.

This module handles the conversion of quantum circuit AST to MLIR
representation using the quantum dialect.
"""

# Import config to set up MLIR path automatically
from . import config  # noqa: F401
from mlir import ir
from .ast import QuantumCircuit


def circuit_to_mlir(circuit: QuantumCircuit, function_name: str = "main") -> str:
    """Convert a quantum circuit to MLIR representation.

    Args:
        circuit: The quantum circuit to convert
        function_name: Name of the generated function

    Returns:
        MLIR string representation of the circuit
    """
    with ir.Context() as ctx, ir.Location.unknown():
        # Allow unregistered dialects for now
        ctx.allow_unregistered_dialects = True
        ctx.enable_multithreading(False)

        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            # Create function type: () -> ()
            func_type = ir.FunctionType.get([], [])

            # Create function using lower-level API
            func_op = ir.Operation.create(
                "func.func",
                attributes={
                    "sym_name": ir.StringAttr.get(function_name),
                    "function_type": ir.TypeAttr.get(func_type),
                },
                regions=1,
            )

            # Get the function body
            func_body = func_op.regions[0].blocks.append()

            with ir.InsertionPoint(func_body):
                # Allocate qubits on demand
                i32_type = ir.IntegerType.get_signless(32)
                ssa_qubits = {}

                def get_ssa(idx):
                    if idx in ssa_qubits:
                        return ssa_qubits[idx]
                    op = ir.Operation.create("quantum.alloc", results=[i32_type], attributes={})
                    ssa_qubits[idx] = op.result
                    return ssa_qubits[idx]

                # Generate quantum operations
                for gate in circuit.gates:
                    if gate.name == "i":
                        ir.Operation.create("quantum.i", operands=[get_ssa(gate.q[0])], attributes={})
                    elif gate.name == "x":
                        ir.Operation.create("quantum.x", operands=[get_ssa(gate.q[0])], attributes={})
                    elif gate.name == "y":
                        ir.Operation.create("quantum.y", operands=[get_ssa(gate.q[0])], attributes={})
                    elif gate.name == "z":
                        ir.Operation.create("quantum.z", operands=[get_ssa(gate.q[0])], attributes={})
                    elif gate.name == "h":
                        ir.Operation.create("quantum.h", operands=[get_ssa(gate.q[0])], attributes={})
                    elif gate.name == "cx":
                        ir.Operation.create(
                            "quantum.cx", operands=[get_ssa(gate.q[0]), get_ssa(gate.q[1])], attributes={}
                        )
                    else:
                        raise ValueError(f"Unknown gate: {gate.name}")

                # Return from function
                ir.Operation.create("func.return", operands=[], attributes={})

        return str(module)
