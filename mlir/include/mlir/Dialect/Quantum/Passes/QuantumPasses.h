//===- QuantumPasses.h - Quantum dialect passes ---------------*- C++ -*-===//

#ifndef MLIR_DIALECT_QUANTUM_PASSES_H
#define MLIR_DIALECT_QUANTUM_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace quantum {

/// Create a pass that cancels adjacent X gates on the same qubit
std::unique_ptr<mlir::Pass> createCancelXPass();

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Quantum/Passes/Passes.h.inc"

} // namespace quantum
} // namespace mlir

#endif // MLIR_DIALECT_QUANTUM_PASSES_H
