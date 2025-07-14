add_mlir_library(MLIRQuantum
  QuantumDialect.cpp
  QuantumOps.cpp
  QuantumCancelXPass.cpp
  init.cpp

  DEPENDS
  MLIRIR
  LINK_LIBS
  PUBLIC
  MLIRIR
)