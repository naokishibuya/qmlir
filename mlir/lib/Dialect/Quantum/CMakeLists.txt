include_directories(${MLIR_MAIN_INCLUDE_DIR})

set(LLVM_TARGET_DEFINITIONS QuantumOps.td)
mlir_tablegen(QuantumOps.h.inc -gen-op-decls)
mlir_tablegen(QuantumOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRQuantumOpsIncGen)

add_mlir_library(MLIRQuantum
  QuantumDialect.cpp
  QuantumOps.cpp
  QuantumCancelXPass.cpp
  init.cpp

  DEPENDS
  MLIRIR
  MLIRQuantumOpsIncGen

  LINK_LIBS
  PUBLIC
  MLIRIR
)
