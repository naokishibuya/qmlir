//===- QuantumOps.td - Quantum dialect ops ---------------*- tablegen -*-===//
//
// Quantum dialect operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Quantum dialect definition.
//===----------------------------------------------------------------------===//

def Quantum_Dialect : Dialect {
  let name = "quantum";
  let cppNamespace = "::mlir::quantum";
  let summary = "A simple quantum computing dialect";
  let description = [{
    This dialect provides basic quantum computing operations.
  }];
}

//===----------------------------------------------------------------------===//
// Base quantum operation definition.
//===----------------------------------------------------------------------===//

class Quantum_Op<string mnemonic, list<Trait> traits = []> : 
    Op<Quantum_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Quantum operations.
//===----------------------------------------------------------------------===//

def Quantum_AllocOp : Quantum_Op<"alloc"> {
  let summary = "Allocate a qubit";
  let description = [{
    Allocates a new qubit.
  }];
  let results = (outs AnyType:$result);
}

def Quantum_IOp : Quantum_Op<"i"> {
  let summary = "Identity gate";
  let description = [{
    Applies an identity gate to a qubit (no-op).
  }];
  let arguments = (ins AnyType:$qubit);
}

def Quantum_XOp : Quantum_Op<"x"> {
  let summary = "Pauli-X gate";
  let description = [{
    Applies a Pauli-X gate to a qubit.
  }];
  let arguments = (ins AnyType:$qubit);
}

def Quantum_YOp : Quantum_Op<"y"> {
  let summary = "Pauli-Y gate";
  let description = [{
    Applies a Pauli-Y gate to a qubit.
  }];
  let arguments = (ins AnyType:$qubit);
}

def Quantum_ZOp : Quantum_Op<"z"> {
  let summary = "Pauli-Z gate";
  let description = [{
    Applies a Pauli-Z gate to a qubit.
  }];
  let arguments = (ins AnyType:$qubit);
}

def Quantum_HOp : Quantum_Op<"h"> {
  let summary = "Hadamard gate";
  let description = [{
    Applies a Hadamard gate to a qubit.
  }];
  let arguments = (ins AnyType:$qubit);
}

def Quantum_CXOp : Quantum_Op<"cx"> {
  let summary = "Controlled-X gate";
  let description = [{
    Applies a controlled-X gate with control and target qubits.
  }];
  let arguments = (ins AnyType:$control, AnyType:$target);
}
