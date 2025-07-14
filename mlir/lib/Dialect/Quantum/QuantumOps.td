//===- QuantumOps.td - Quantum Dialect Ops ----------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/IR/Traits.td"

// Dialect ------------------------------------------------------------------
def Quantum_Dialect : Dialect {
  let name = "quantum";
  let cppNamespace = "::mlir::quantum";
}

// Qubit type ---------------------------------------------------------------
def Quantum_QubitType : TypeDef<Quantum_Dialect, "Qubit"> {
  let mnemonic     = "qubit";
  let summary      = "Quantum qubit type";
  let description  = [{Represents a single qubit in the quantum dialect.}];
}

// Common op base -----------------------------------------------------------
class Quantum_Op<string mnemonic> : Op<Quantum_Dialect, mnemonic>;

// Gates --------------------------------------------------------------------
def Quantum_XOp : Quantum_Op<"x"> {
  let arguments    = (ins Quantum_QubitType:$operand);
  let summary      = "Pauli-X gate";
}
def Quantum_HOp : Quantum_Op<"h"> {
  let arguments    = (ins Quantum_QubitType:$operand);
  let summary      = "Hadamard gate";
}
def Quantum_CXOp : Quantum_Op<"cx"> {
  let arguments    = (ins Quantum_QubitType:$ctrl, Quantum_QubitType:$target);
  let summary      = "Controlled-X gate";
}

// Allocation ---------------------------------------------------------------
def Quantum_AllocOp : Op<Quantum_Dialect, "alloc"> {
  let results      = (outs Quantum_QubitType:$qubit);
  let summary      = "Qubit allocation";
}
