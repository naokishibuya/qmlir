//===- QuantumCancelXPass.cpp - Quantum X Gate Cancellation Pass --------===//

#include "Quantum/QuantumOps.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct QuantumCancelXPass
    : public PassWrapper<QuantumCancelXPass, OperationPass<mlir::FuncOp>> {
  void runOnOperation() override {
    auto func = getOperation();
    func.walk([&](quantum::XOp op) {
      if (auto nextOp = op->getNextNode()) {
        if (auto nextX = dyn_cast<quantum::XOp>(nextOp)) {
          if (op.getOperand() == nextX.getOperand()) {
            op.erase();
            nextX.erase();
          }
        }
      }
    });
  }
};
} // end anonymous namespace

namespace mlir {
void registerQuantumCancelXPass() {
  PassRegistration<QuantumCancelXPass>(
      "quantum-cancel-x",
      "Cancel consecutive quantum.x gates on the same qubit");
}
} // namespace mlir
