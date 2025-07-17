//===- Passes.td - Quantum dialect passes ----------------*- tablegen -*-===//

include "mlir/Pass/PassBase.td"

def CancelSelfInversePass : Pass<"quantum-cancel-self-inverse", "func::FuncOp"> {
  let summary = "Cancel adjacent self-inverse quantum gates";
  let description = [{
    This pass cancels adjacent self-inverse quantum gates (X, H, Z, etc.) 
    operating on the same qubit, since Op * Op = I for involutory operations.
    
    Example:
    ```mlir
    quantum.x %q : i32
    quantum.x %q : i32
    quantum.h %q : i32
    quantum.h %q : i32
    ```
    becomes empty (all operations are removed).
  }];
  
  let constructor = "mlir::quantum::createCancelSelfInversePass()";
}
