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

// -----------------------------------------------------------------------------
// Custom traits
// -----------------------------------------------------------------------------

def SelfInverse : NativeOpTrait<"SelfInverse"> {
  let cppNamespace = "::mlir::quantum";
}

def Hermitian : NativeOpTrait<"Hermitian"> {
  let cppNamespace = "::mlir::quantum";
}

//===----------------------------------------------------------------------===//
// Base quantum operation definition.
//===----------------------------------------------------------------------===//

class Quantum_Op<string mnemonic, list<Trait> traits = []> :
    Op<Quantum_Dialect, mnemonic, traits>;

def Quantum_AllocOp : Quantum_Op<"alloc"> {
  let summary = "Allocate a qubit";
  let description = [{
    Allocates a new qubit.
  }];
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Single-qubit gates.
//===----------------------------------------------------------------------===//

def Quantum_IOp : Quantum_Op<"i", [SelfInverse, Hermitian]> {
  let summary = "Identity gate";
  let description = [{
    Applies an identity gate to a qubit (no-op).
  }];
  let arguments = (ins AnyType:$qubit);
}

def Quantum_XOp : Quantum_Op<"x", [SelfInverse, Hermitian]> {
  let summary = "Pauli-X gate";
  let description = [{
    Applies a Pauli-X gate to a qubit.
  }];
  let arguments = (ins AnyType:$qubit);
}

def Quantum_YOp : Quantum_Op<"y", [SelfInverse, Hermitian]> {
  let summary = "Pauli-Y gate";
  let description = [{
    Applies a Pauli-Y gate to a qubit.
  }];
  let arguments = (ins AnyType:$qubit);
}

def Quantum_ZOp : Quantum_Op<"z", [SelfInverse, Hermitian]> {
  let summary = "Pauli-Z gate";
  let description = [{
    Applies a Pauli-Z gate to a qubit.
  }];
  let arguments = (ins AnyType:$qubit);
}

def Quantum_HOp : Quantum_Op<"h", [SelfInverse, Hermitian]> {
  let summary = "Hadamard gate";
  let description = [{
    Applies a Hadamard gate to a qubit.
  }];
  let arguments = (ins AnyType:$qubit);
}

def Quantum_SOp : Quantum_Op<"s"> {
  let summary = "S gate (Phase gate)";
  let description = [{
    Applies an S gate (phase gate) to a qubit.
    S = [[1, 0], [0, i]] = RZ(π/2)
  }];
  let arguments = (ins AnyType:$qubit);
}

def Quantum_TOp : Quantum_Op<"t"> {
  let summary = "T gate (π/8 gate)";
  let description = [{
    Applies a T gate (π/8 gate) to a qubit.
    T = [[1, 0], [0, e^(iπ/4)]] = RZ(π/4)
  }];
  let arguments = (ins AnyType:$qubit);
}

def Quantum_SdgOp : Quantum_Op<"sdg"> {
  let summary = "S-dagger gate";
  let description = [{
    Applies an S-dagger gate to a qubit.
    S† = [[1, 0], [0, -i]] = RZ(-π/2)
  }];
  let arguments = (ins AnyType:$qubit);
}

def Quantum_TdgOp : Quantum_Op<"tdg"> {
  let summary = "T-dagger gate";
  let description = [{
    Applies a T-dagger gate to a qubit.
    T† = [[1, 0], [0, e^(-iπ/4)]] = RZ(-π/4)
  }];
  let arguments = (ins AnyType:$qubit);
}

//===----------------------------------------------------------------------===//
// Multi-qubit gates.
//===----------------------------------------------------------------------===//

def Quantum_CXOp : Quantum_Op<"cx", [SelfInverse, Hermitian]> {
  let summary = "Controlled-X gate";
  let description = [{
    Applies a controlled-X gate with control and target qubits.
  }];
  let arguments = (ins AnyType:$control, AnyType:$target);
}

def Quantum_CYOp : Quantum_Op<"cy", [SelfInverse, Hermitian]> {
  let summary = "Controlled-Y gate";
  let description = [{
    Applies a controlled-Y gate with control and target qubits.
  }];
  let arguments = (ins AnyType:$control, AnyType:$target);
}

def Quantum_CZOp : Quantum_Op<"cz", [SelfInverse, Hermitian]> {
  let summary = "Controlled-Z gate";
  let description = [{
    Applies a controlled-Z gate with control and target qubits.
  }];
  let arguments = (ins AnyType:$control, AnyType:$target);
}

//===----------------------------------------------------------------------===//
// Rotation gates.
//===----------------------------------------------------------------------===//

def Quantum_RXOp : Quantum_Op<"rx"> {
  let summary = "Rotation around X-axis";
  let description = [{
    Applies a rotation around the X-axis by the given angle.
    RX(θ) = exp(-i * θ/2 * X)
  }];
  let arguments = (ins AnyType:$qubit, F64:$angle);
}

def Quantum_RYOp : Quantum_Op<"ry"> {
  let summary = "Rotation around Y-axis";
  let description = [{
    Applies a rotation around the Y-axis by the given angle.
    RY(θ) = exp(-i * θ/2 * Y)
  }];
  let arguments = (ins AnyType:$qubit, F64:$angle);
}

def Quantum_RZOp : Quantum_Op<"rz"> {
  let summary = "Rotation around Z-axis";
  let description = [{
    Applies a rotation around the Z-axis by the given angle.
    RZ(θ) = exp(-i * θ/2 * Z)
  }];
  let arguments = (ins AnyType:$qubit, F64:$angle);
}
