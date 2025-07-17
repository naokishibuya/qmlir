add_executable(quantum-opt
  quantum-opt.cpp
)

llvm_update_compile_flags(quantum-opt)

target_link_libraries(quantum-opt PRIVATE
  MLIRQuantumPasses
  MLIROptLib
  MLIRParser
)

mlir_check_all_link_libraries(quantum-opt)
