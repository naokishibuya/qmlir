//===- QuantumOps.td - Quantum Dialect Ops ---------------------*- tablegen -*-===//
//
// Minimal Quantum Dialect for MLIR
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def Quantum_Dialect : Dialect {
  let name = "quantum";
  let cppNamespace = "::mlir::quantum";
}

class Quantum_Op<string mnemonic> : Op<Dialect = Quantum_Dialect, mnemonic> {
  let assemblyFormat = "$operand attr-dict";
  let arguments = (ins AnyType:$operand);
}

def Quantum_XOp : Quantum_Op<"x">;
def Quantum_HOp : Quantum_Op<"h">;
def Quantum_CXOp : Op<Quantum_Dialect, "cx", [NoSideEffect]> {
  let assemblyFormat = "$ctrl, $target attr-dict";
  let arguments = (ins AnyType:$ctrl, AnyType:$target);
}

def Quantum_AllocOp : Op<Quantum_Dialect, "alloc", [NoSideEffect]> {
  let results = (outs AnyType);
  let assemblyFormat = attr-dict;
}
