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

/// Generic pattern to cancel involutory (self-inverse) operations on
/// the same qubit, even when separated by operations on other qubits.
/// Involutory gates satisfy: G * G = I (identity) Examples: X, Y, Z, H, I
template <typename OpType>
struct CancelInvolutoryPattern : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    Value qubit = op.getQubit();

    // Look for the next operation of the same type on the same qubit
    Operation *nextOp = op.getOperation()->getNextNode();
    while (nextOp) {
      // Check if it's the same operation type
      if (auto nextSameOp = dyn_cast<OpType>(nextOp)) {
        // Check if they operate on the same qubit
        if (nextSameOp.getQubit() == qubit) {
          // Remove both operations (involutory: Op * Op = I)
          rewriter.eraseOp(nextSameOp);
          rewriter.eraseOp(op);
          return success();
        }
      }

      // Check if this operation interferes with our qubit
      // For single-qubit operations, only operations on the same qubit
      // interfere
      if (auto singleQubitOp = dyn_cast<IOp>(nextOp)) {
        if (singleQubitOp.getQubit() == qubit) {
          return failure();
        }
      } else if (auto singleQubitOp = dyn_cast<XOp>(nextOp)) {
        if (singleQubitOp.getQubit() == qubit) {
          return failure();
        }
      } else if (auto singleQubitOp = dyn_cast<YOp>(nextOp)) {
        if (singleQubitOp.getQubit() == qubit) {
          return failure();
        }
      } else if (auto singleQubitOp = dyn_cast<ZOp>(nextOp)) {
        if (singleQubitOp.getQubit() == qubit) {
          return failure();
        }
      } else if (auto singleQubitOp = dyn_cast<HOp>(nextOp)) {
        if (singleQubitOp.getQubit() == qubit) {
          return failure();
        }
      } else if (auto cxOp = dyn_cast<CXOp>(nextOp)) {
        if (cxOp.getControl() == qubit || cxOp.getTarget() == qubit) {
          // CX operation involves our qubit - can't cancel
          return failure();
        }
      }

      nextOp = nextOp->getNextNode();
    }

    return failure();
  }
};

/// Specialized pattern for CX gates (two-qubit operations)
struct CancelCXPattern : public OpRewritePattern<CXOp> {
  using OpRewritePattern<CXOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CXOp cxOp,
                                PatternRewriter &rewriter) const override {
    Value control = cxOp.getControl();
    Value target = cxOp.getTarget();

    // Look for the next CX operation with the same control and target
    Operation *nextOp = cxOp.getOperation()->getNextNode();
    while (nextOp) {
      // Check if it's another CX operation
      if (auto nextCXOp = dyn_cast<CXOp>(nextOp)) {
        // Check if they have the same control and target qubits
        if (nextCXOp.getControl() == control &&
            nextCXOp.getTarget() == target) {
          // Remove both operations (CX * CX = I)
          rewriter.eraseOp(nextCXOp);
          rewriter.eraseOp(cxOp);
          return success();
        }
      }

      // Check if this operation interferes with our control or target qubits
      if (auto singleQubitOp = dyn_cast<IOp>(nextOp)) {
        if (singleQubitOp.getQubit() == control ||
            singleQubitOp.getQubit() == target) {
          return failure();
        }
      } else if (auto singleQubitOp = dyn_cast<XOp>(nextOp)) {
        if (singleQubitOp.getQubit() == control ||
            singleQubitOp.getQubit() == target) {
          return failure();
        }
      } else if (auto singleQubitOp = dyn_cast<YOp>(nextOp)) {
        if (singleQubitOp.getQubit() == control ||
            singleQubitOp.getQubit() == target) {
          return failure();
        }
      } else if (auto singleQubitOp = dyn_cast<ZOp>(nextOp)) {
        if (singleQubitOp.getQubit() == control ||
            singleQubitOp.getQubit() == target) {
          return failure();
        }
      } else if (auto singleQubitOp = dyn_cast<HOp>(nextOp)) {
        if (singleQubitOp.getQubit() == control ||
            singleQubitOp.getQubit() == target) {
          return failure();
        }
      } else if (auto otherCXOp = dyn_cast<CXOp>(nextOp)) {
        if (otherCXOp.getControl() == control ||
            otherCXOp.getControl() == target ||
            otherCXOp.getTarget() == control ||
            otherCXOp.getTarget() == target) {
          return failure();
        }
      }

      nextOp = nextOp->getNextNode();
    }

    return failure();
  }
};

void CancelSelfInversePass::runOnOperation() {
  auto func = getOperation();

  RewritePatternSet patterns(&getContext());

  // Add patterns for all involutory (self-inverse) gates
  patterns.add<CancelInvolutoryPattern<IOp>>(&getContext());
  patterns.add<CancelInvolutoryPattern<XOp>>(&getContext());
  patterns.add<CancelInvolutoryPattern<YOp>>(&getContext());
  patterns.add<CancelInvolutoryPattern<ZOp>>(&getContext());
  patterns.add<CancelInvolutoryPattern<HOp>>(&getContext());
  patterns.add<CancelCXPattern>(&getContext());
  // Add more involutory gates here as needed

  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<mlir::Pass> mlir::quantum::createCancelSelfInversePass() {
  return std::make_unique<CancelSelfInversePass>();
}
