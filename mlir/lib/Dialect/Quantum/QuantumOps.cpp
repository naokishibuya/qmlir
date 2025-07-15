//===- QuantumOps.cpp - Quantum dialect ops -------------------*- C++ -*-===//

#include "mlir/Dialect/Quantum/IR/Quantum.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::quantum;

#include "mlir/Dialect/Quantum/IR/QuantumOpsDialect.cpp.inc"

void QuantumDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Quantum/IR/QuantumOps.cpp.inc"
  >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Quantum/IR/QuantumOps.cpp.inc"
