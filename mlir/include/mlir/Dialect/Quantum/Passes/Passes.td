//===- Passes.td - Quantum dialect passes ----------------*- tablegen -*-===//

include "mlir/Pass/PassBase.td"

def CancelXPass : Pass<"quantum-cancel-x", "func::FuncOp"> {
  let summary = "Cancel adjacent X gates on the same qubit";
  let description = [{
    This pass removes adjacent X operations on the same qubit, since X*X = I.
    
    Example:
    ```mlir
    quantum.x %q : i32
    quantum.x %q : i32
    ```
    becomes empty (both operations are removed).
  }];
  
  let constructor = "mlir::quantum::createCancelXPass()";
}
