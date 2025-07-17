//===- quantum-opt.cpp - Quantum dialect optimization tool ----*- C++ -*-===//
//
// Simple optimization tool for Quantum dialect
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quantum/IR/Quantum.h"
#include "mlir/Dialect/Quantum/Passes/QuantumPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

int main(int argc, char **argv) {
  // Register quantum dialect and passes
  mlir::DialectRegistry registry;
  registry.insert<mlir::quantum::QuantumDialect>();
  registry.insert<mlir::func::FuncDialect>();
  mlir::quantum::registerQuantumPasses();
  
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Quantum optimizer\n", registry));
}
