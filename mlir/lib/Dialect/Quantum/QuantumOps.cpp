//===- QuantumOps.cpp - Quantum Dialect Ops ---------------------===//

#include "Quantum/QuantumOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::quantum;

#define GET_OP_CLASSES
#include "Quantum/QuantumOps.cpp.inc"
