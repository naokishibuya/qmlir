"""
Translate a toy Python circuit into our Quantum Dialect MLIR using the official
MLIR Python bindings.  No string‑splicing — we build IR programmatically.

Usage:
  python python/emit_quantum_mlir.py | ./build/bin/test-quantum-dialect
  python python/emit_quantum_mlir.py > test.mlir && mlir-opt test.mlir --quantum-cancel-x

If you compiled with `-DMLIR_ENABLE_BINDINGS_PYTHON=ON` and installed the
wheel, this will work *out of the box*.  Otherwise, make sure the wheel is on
`PYTHONPATH` and that `libMLIRQuantum` is pre‑loaded (see README notes).
"""

from mlir import ir
import ctypes
import os
import platform


# -----------------------------------------------------------------------------
# Load our Quantum plugin so the dialect & pass register with the global tables
# -----------------------------------------------------------------------------
lib_name = {
    "Darwin": "libMLIRQuantum.dylib",
    "Windows": "MLIRQuantum.dll", 
}.get(platform.system(), "libMLIRQuantum.so")

try:
    ctypes.CDLL(lib_name)  # Rely on system ld search paths
except OSError:
    # Fallback: look in build/lib directory
    here = os.path.abspath(os.path.dirname(__file__))
    candidate = os.path.join(here, "..", "build", "lib", lib_name)
    try:
        ctypes.CDLL(candidate)
    except OSError:
        # Another fallback: look in standard lib directory
        candidate = os.path.join(here, "..", "lib", lib_name)
        ctypes.CDLL(candidate)


# -----------------------------------------------------------------------------
# Toy front‑end: ultra‑simple circuit DSL
# -----------------------------------------------------------------------------
class Gate:
    def __init__(self, name, *q):
        self.name = name
        self.q = q

class Circuit:
    def __init__(self):
        self.gates = []

    def x(self, q):
        self.gates.append(Gate("x", q))

    def h(self, q):
        self.gates.append(Gate("h", q))

    def cx(self, c, t):
        self.gates.append(Gate("cx", c, t))


# -----------------------------------------------------------------------------
# Emit MLIR using the Python bindings
# -----------------------------------------------------------------------------
def circuit_to_mlir(circ: Circuit) -> ir.Module:
    with ir.Context() as ctx, ir.Location.unknown():
        # Until we have generated Python stubs, allow unregistered dialects.
        ctx.allow_unregistered_dialects = True
        # Register the quantum dialect
        ctx.enable_multithreading(False)
        
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            # Create a function to contain our quantum operations
            # Our quantum dialect uses i32 for qubit references
            i32_type = ir.IntegerType.get_signless(32)
            func_type = ir.FunctionType.get([], [])
            
            func_op = ir.Operation.create(
                "func.func",
                attributes={
                    "sym_name": ir.StringAttr.get("quantum_circuit"),
                    "function_type": ir.TypeAttr.get(func_type),
                },
                regions=1
            )
            
            # Get the function body
            func_body = func_op.regions[0].blocks.append()
            
            with ir.InsertionPoint(func_body):
                # Allocate one qubit per logical id we encounter on the fly.
                ssa_qubits = {}

                def get_ssa(idx):
                    if idx in ssa_qubits:
                        return ssa_qubits[idx]
                    op = ir.Operation.create(
                        "quantum.alloc",
                        results=[i32_type],
                        attributes={}
                    )
                    ssa_qubits[idx] = op.result
                    return ssa_qubits[idx]

                for g in circ.gates:
                    if g.name == "x":
                        ir.Operation.create(
                            "quantum.x",
                            operands=[get_ssa(g.q[0])],
                            attributes={}
                        )
                    elif g.name == "h":
                        ir.Operation.create(
                            "quantum.h",
                            operands=[get_ssa(g.q[0])],
                            attributes={}
                        )
                    elif g.name == "cx":
                        ir.Operation.create(
                            "quantum.cx",
                            operands=[get_ssa(g.q[0]), get_ssa(g.q[1])],
                            attributes={}
                        )
                    else:
                        raise ValueError(f"unsupported gate {g.name}")

                # Add return statement
                ir.Operation.create("func.return", attributes={})

        return module


def main():
    # Example circuit: X; X; H on qubit 0
    circ = Circuit()
    circ.x(0)
    circ.x(0)
    circ.h(0)

    mod = circuit_to_mlir(circ)
    print(mod)

if __name__ == "__main__":
    main()
