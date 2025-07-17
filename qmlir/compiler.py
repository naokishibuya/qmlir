"""MLIR Code Generation for Quantum Circuits

This module handles the conversion of quantum circuit representation to MLIR
representation using the quantum dialect.
"""

# Import backend to set up MLIR path automatically
from . import backend  # noqa: F401
from mlir import ir
from .circuit import QuantumCircuit


def circuit_to_mlir(circuit: QuantumCircuit, function_name: str = "main") -> str:
    """Convert a quantum circuit to MLIR representation.

    Args:
        circuit: The quantum circuit to convert
        function_name: Name of the generated function

    Returns:
        MLIR string representation of the circuit
    """
    # Phase 1: Collect unique parameters using simplified approach
    param_index_map = {}
    for gate in circuit.gates:
        for param in gate.parameters:
            if param.id not in param_index_map:
                param_index_map[param.id] = len(param_index_map)

    with ir.Context() as ctx, ir.Location.unknown():
        # Allow unregistered dialects for now
        ctx.allow_unregistered_dialects = True
        ctx.enable_multithreading(False)

        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            # Create function type with parameters
            f64_type = ir.F64Type.get()
            func_args = [f64_type] * len(param_index_map)
            func_type = ir.FunctionType.get(func_args, [])

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

            # Add function arguments to the block
            for _ in range(len(param_index_map)):
                func_body.add_argument(f64_type, ir.Location.unknown())

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
                    elif gate.name == "s":
                        ir.Operation.create("quantum.s", operands=[get_ssa(gate.q[0])], attributes={})
                    elif gate.name == "t":
                        ir.Operation.create("quantum.t", operands=[get_ssa(gate.q[0])], attributes={})
                    elif gate.name == "sdg":
                        ir.Operation.create("quantum.sdg", operands=[get_ssa(gate.q[0])], attributes={})
                    elif gate.name == "tdg":
                        ir.Operation.create("quantum.tdg", operands=[get_ssa(gate.q[0])], attributes={})
                    elif gate.name == "cx":
                        ir.Operation.create(
                            "quantum.cx", operands=[get_ssa(gate.q[0]), get_ssa(gate.q[1])], attributes={}
                        )
                    elif gate.name == "cy":
                        ir.Operation.create(
                            "quantum.cy", operands=[get_ssa(gate.q[0]), get_ssa(gate.q[1])], attributes={}
                        )
                    elif gate.name == "cz":
                        ir.Operation.create(
                            "quantum.cz", operands=[get_ssa(gate.q[0]), get_ssa(gate.q[1])], attributes={}
                        )
                    # Rotation gates
                    elif gate.name == "rx":
                        param_idx = param_index_map[gate.parameters[0].id]
                        param_value = func_body.arguments[param_idx]
                        # For now, generate a placeholder comment
                        ir.Operation.create("quantum.rx", operands=[get_ssa(gate.q[0]), param_value], attributes={})
                    elif gate.name == "ry":
                        param_idx = param_index_map[gate.parameters[0].id]
                        param_value = func_body.arguments[param_idx]
                        ir.Operation.create("quantum.ry", operands=[get_ssa(gate.q[0]), param_value], attributes={})
                    elif gate.name == "rz":
                        param_idx = param_index_map[gate.parameters[0].id]
                        param_value = func_body.arguments[param_idx]
                        ir.Operation.create("quantum.rz", operands=[get_ssa(gate.q[0]), param_value], attributes={})
                    else:
                        raise ValueError(f"Unknown gate: {gate.name}")

                # Return from function
                ir.Operation.create("func.return", operands=[], attributes={})

        return str(module)
