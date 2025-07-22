//===- QuantumCancelSelfInversePass.cpp - Cancel self-inverse gates -------===//
//
// This pass cancels pairs of self-inverse gates (e.g., X X → I) when they act
// on the same qubits and are not interrupted by interfering gates.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quantum/IR/Quantum.h"
#include "mlir/Dialect/Quantum/Passes/QuantumPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::quantum;

namespace {
#define GEN_PASS_DEF_CANCELSELFINVERSEPASS
#include "mlir/Dialect/Quantum/Passes/Passes.h.inc"

struct CancelSelfInversePass
    : public impl::CancelSelfInversePassBase<CancelSelfInversePass> {
  void runOnOperation() override;
};

/// Generic pattern to cancel self-inverse operations.
/// Works for any op with the SelfInverse trait.
struct CancelGenericSelfInversePattern : public RewritePattern {
  CancelGenericSelfInversePattern(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Only match ops with the SelfInverse trait
    if (!op->hasTrait<mlir::quantum::SelfInverse>())
      return failure();

    auto operands = op->getOperands();
    Operation *nextOp = op->getNextNode();

    while (nextOp) {
      // Skip non-identical ops
      if (nextOp->getName() == op->getName() &&
          nextOp->getOperands() == operands) {
        // Cancel both ops
        rewriter.eraseOp(nextOp);
        rewriter.eraseOp(op);
        return success();
      }

      // If any of the same operands are used, cancel attempt fails
      for (Value operand : operands) {
        if (llvm::is_contained(nextOp->getOperands(), operand))
          return failure(); // Interfering op found
      }

      nextOp = nextOp->getNextNode();
    }

    return failure();
  }
};

void CancelSelfInversePass::runOnOperation() {
  auto func = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<CancelGenericSelfInversePattern>(&getContext());

  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<mlir::Pass> mlir::quantum::createCancelSelfInversePass() {
  return std::make_unique<CancelSelfInversePass>();
}
