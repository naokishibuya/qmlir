#include "mlir/Dialect/Quantum/IR/Quantum.h"

using namespace mlir;
using namespace mlir::quantum;

#include "mlir/Dialect/Quantum/IR/QuantumOpsDialect.cpp.inc"

void QuantumDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Quantum/IR/QuantumOps.cpp.inc"
  >();
}
