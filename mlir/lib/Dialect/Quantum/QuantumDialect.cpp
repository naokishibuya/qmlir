//===- QuantumDialect.cpp - Quantum Dialect Definition ------------------===//

#include "mlir/Dialect/Quantum/QuantumDialect.h"
#include "mlir/Dialect/Quantum/QuantumOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::quantum;

void QuantumDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Quantum/QuantumOps.cpp.inc"
  >();
}

QuantumDialect::QuantumDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<QuantumDialect>()) {
  initialize();
}