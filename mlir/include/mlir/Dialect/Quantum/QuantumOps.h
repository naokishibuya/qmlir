//===- QuantumOps.h - Quantum Dialect Ops ------------------------*- C++ -*-===//

#ifndef QUANTUM_OPS_H
#define QUANTUM_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Dialect/Quantum/QuantumOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Quantum/QuantumOps.h.inc"

#endif // QUANTUM_OPS_H
