//===- Quantum.h - Quantum dialect ----------------*- C++ -*-===//
//
// Quantum dialect for MLIR
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_QUANTUM_IR_QUANTUM_H
#define MLIR_DIALECT_QUANTUM_IR_QUANTUM_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Dialect/Quantum/IR/QuantumTraits.h"

#include "mlir/Dialect/Quantum/IR/QuantumOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Quantum/IR/QuantumOps.h.inc"

#endif // MLIR_DIALECT_QUANTUM_IR_QUANTUM_H
