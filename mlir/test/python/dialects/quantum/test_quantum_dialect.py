# RUN: %PYTHON %s | FileCheck %s

import os
import sys

# Auto-detect the MLIR Python bindings path
script_dir = os.path.dirname(os.path.abspath(__file__))
current = script_dir
while current != '/':
    build_dir = os.path.join(current, 'build')
    mlir_python_path = os.path.join(build_dir, 'tools', 'mlir', 'python_packages', 'mlir_core')
    if os.path.exists(mlir_python_path):
        sys.path.insert(0, mlir_python_path)
        break
    current = os.path.dirname(current)

from mlir import ir

# Simple test: Create a basic quantum circuit and verify the output
with ir.Context() as ctx, ir.Location.unknown():
    ctx.allow_unregistered_dialects = True
    
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
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
        
        func_body = func_op.regions[0].blocks.append()
        
        with ir.InsertionPoint(func_body):
            # Create X; X; H circuit
            qubit = ir.Operation.create("quantum.alloc", results=[i32_type]).result
            ir.Operation.create("quantum.x", operands=[qubit])
            ir.Operation.create("quantum.x", operands=[qubit])  
            ir.Operation.create("quantum.h", operands=[qubit])
            ir.Operation.create("func.return")

    print("Generated MLIR:")
    print(module)

# CHECK: Generated MLIR:
# CHECK: module {
# CHECK:   func.func @quantum_circuit() {
# CHECK:     %{{.*}} = "quantum.alloc"() : () -> i32
# CHECK:     "quantum.x"(%{{.*}}) : (i32) -> ()
# CHECK:     "quantum.x"(%{{.*}}) : (i32) -> ()
# CHECK:     "quantum.h"(%{{.*}}) : (i32) -> ()
# CHECK:     return
# CHECK:   }
# CHECK: }
