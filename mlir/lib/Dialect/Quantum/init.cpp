//===- init.cpp - Quantum Dialect Registration -----------------*- C++ -*-===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "Quantum/QuantumDialect.h"

using namespace mlir;
using namespace mlir::quantum;

void registerQuantumDialect(DialectRegistry &registry) {
  registry.insert<QuantumDialect>();
}

namespace mlir {
void registerQuantumCancelXPass();
}

extern "C" void registerQuantumPasses() {
  mlir::registerQuantumCancelXPass();
}
