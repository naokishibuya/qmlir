//===- test-quantum.cpp - Test quantum dialect ----------*- C++ -*-===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quantum/IR/Quantum.h"
#include "mlir/Dialect/Quantum/Passes/QuantumPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <string>

using namespace mlir;

int main(int argc, char **argv) {
  MLIRContext context;

  // Register the quantum and func dialects
  context.getOrLoadDialect<quantum::QuantumDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  std::string mlirContent;

  if (argc == 1) {
    // Read from stdin (pipe support)
    std::string line;
    while (std::getline(std::cin, line)) {
      mlirContent += line + "\n";
    }
    if (mlirContent.empty()) {
      llvm::errs() << "Usage: " << argv[0] << " <input-file>\n";
      llvm::errs() << "   or: cat <input-file> | " << argv[0] << "\n";
      return 1;
    }
  } else if (argc == 2) {
    // Read from file
    std::string errorMessage;
    auto fileOrErr = openInputFile(argv[1], &errorMessage);
    if (!fileOrErr) {
      llvm::errs() << "Error opening file: " << errorMessage << "\n";
      return 1;
    }
    mlirContent = (*fileOrErr).getBuffer().str();
  } else {
    llvm::errs() << "Usage: " << argv[0] << " <input-file>\n";
    llvm::errs() << "   or: cat <input-file> | " << argv[0] << "\n";
    return 1;
  }

  // Parse the MLIR from content
  auto module = parseSourceString<ModuleOp>(mlirContent, &context);

  if (!module) {
    llvm::errs() << "Failed to parse MLIR!\n";
    return 1;
  }

  llvm::outs() << "=== Original MLIR ===\n";
  module->print(llvm::outs());

  // Apply quantum optimization passes
  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(quantum::createCancelXPass());

  if (failed(pm.run(*module))) {
    llvm::errs() << "Failed to run optimization passes!\n";
    return 1;
  }

  llvm::outs() << "\n=== Optimized MLIR ===\n";
  module->print(llvm::outs());

  return 0;
}
