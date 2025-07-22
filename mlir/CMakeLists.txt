# MLIR Quantum Dialect

add_subdirectory(include/mlir/Dialect/Quantum)
add_subdirectory(lib/Dialect/Quantum)
add_subdirectory(tools)

set(LLVM_LIT_EXECUTABLE "${LLVM_BINARY_DIR}/bin/llvm-lit")

add_custom_target(check-quantum
  COMMAND ${LLVM_LIT_EXECUTABLE} -sv ${CMAKE_CURRENT_SOURCE_DIR}/test/Dialect/Quantum
  DEPENDS quantum-opt
  COMMENT "Running Quantum dialect tests"
)
