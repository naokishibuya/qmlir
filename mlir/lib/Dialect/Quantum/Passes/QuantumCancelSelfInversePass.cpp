//===- QuantumCancelSelfInversePass.cpp - Cancel self-inverse gates -*- C++
//-*-===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quantum/IR/Quantum.h"
#include "mlir/Dialect/Quantum/Passes/QuantumPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace quantum {
#define GEN_PASS_DEF_CANCELSELFINVERSEPASS
#include "mlir/Dialect/Quantum/Passes/Passes.h.inc"
} // namespace quantum
} // namespace mlir

using namespace mlir;
using namespace mlir::quantum;

namespace {

struct CancelSelfInversePass
    : public mlir::quantum::impl::CancelSelfInversePassBase<
          CancelSelfInversePass> {
  void runOnOperation() override;
};

/// Generic pattern to cancel adjacent involutory (self-inverse) operations on
/// the same qubit Involutory gates satisfy: G * G = I (identity) Examples: X,
/// Y, Z, H, I
template <typename OpType>
struct CancelAdjacentInvolutoryPattern : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    // Get the next operation
    Operation *nextOp = op.getOperation()->getNextNode();
    if (!nextOp)
      return failure();

    // Check if it's the same operation type
    auto nextSameOp = dyn_cast<OpType>(nextOp);
    if (!nextSameOp)
      return failure();

    // Check if they operate on the same qubit
    if (op.getQubit() != nextSameOp.getQubit())
      return failure();

    // Remove both operations (involutory: Op * Op = I)
    rewriter.eraseOp(nextSameOp);
    rewriter.eraseOp(op);

    return success();
  }
};

/// Specialized pattern for CX gates (two-qubit operations)
struct CancelAdjacentCXPattern : public OpRewritePattern<CXOp> {
  using OpRewritePattern<CXOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CXOp cxOp,
                                PatternRewriter &rewriter) const override {
    // Get the next operation
    Operation *nextOp = cxOp.getOperation()->getNextNode();
    if (!nextOp)
      return failure();

    // Check if it's another CX operation
    auto nextCXOp = dyn_cast<CXOp>(nextOp);
    if (!nextCXOp)
      return failure();

    // Check if they have the same control and target qubits
    if (cxOp.getControl() != nextCXOp.getControl() ||
        cxOp.getTarget() != nextCXOp.getTarget())
      return failure();

    // Remove both operations (CX * CX = I)
    rewriter.eraseOp(nextCXOp);
    rewriter.eraseOp(cxOp);

    return success();
  }
};

void CancelSelfInversePass::runOnOperation() {
  auto func = getOperation();

  RewritePatternSet patterns(&getContext());

  // Add patterns for all involutory (self-inverse) gates
  patterns.add<CancelAdjacentInvolutoryPattern<IOp>>(&getContext());
  patterns.add<CancelAdjacentInvolutoryPattern<XOp>>(&getContext());
  patterns.add<CancelAdjacentInvolutoryPattern<YOp>>(&getContext());
  patterns.add<CancelAdjacentInvolutoryPattern<ZOp>>(&getContext());
  patterns.add<CancelAdjacentInvolutoryPattern<HOp>>(&getContext());
  patterns.add<CancelAdjacentCXPattern>(&getContext());
  // Add more involutory gates here as needed

  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<mlir::Pass> mlir::quantum::createCancelSelfInversePass() {
  return std::make_unique<CancelSelfInversePass>();
}
