//===- QuantumCancelXPass.cpp - Cancel adjacent X gates --------*- C++ -*-===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quantum/IR/Quantum.h"
#include "mlir/Dialect/Quantum/Passes/QuantumPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace quantum {
#define GEN_PASS_DEF_CANCELXPASS
#include "mlir/Dialect/Quantum/Passes/Passes.h.inc"
} // namespace quantum
} // namespace mlir

using namespace mlir;
using namespace mlir::quantum;

namespace {

struct CancelXPass : public mlir::quantum::impl::CancelXPassBase<CancelXPass> {
  void runOnOperation() override;
};

/// Pattern to cancel adjacent X operations on the same qubit
struct CancelAdjacentXPattern : public OpRewritePattern<XOp> {
  using OpRewritePattern<XOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(XOp xOp,
                                PatternRewriter &rewriter) const override {
    // Get the next operation
    Operation *nextOp = xOp.getOperation()->getNextNode();
    if (!nextOp)
      return failure();

    // Check if it's another X operation
    auto nextXOp = dyn_cast<XOp>(nextOp);
    if (!nextXOp)
      return failure();

    // Check if they operate on the same qubit
    if (xOp.getQubit() != nextXOp.getQubit())
      return failure();

    // Remove both operations (X * X = I)
    rewriter.eraseOp(nextXOp);
    rewriter.eraseOp(xOp);

    return success();
  }
};

void CancelXPass::runOnOperation() {
  auto func = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<CancelAdjacentXPattern>(&getContext());

  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<mlir::Pass> mlir::quantum::createCancelXPass() {
  return std::make_unique<CancelXPass>();
}
