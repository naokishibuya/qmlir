//===- QuantumDialect.h - Quantum Dialect ----------------------*- C++ -*-===//

#ifndef QUANTUM_DIALECT_H
#define QUANTUM_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace quantum {

class QuantumDialect : public mlir::Dialect {
public:
  explicit QuantumDialect(mlir::MLIRContext *context);
  void initialize() override;
};

} // namespace quantum
} // namespace mlir

#endif // QUANTUM_DIALECT_H
