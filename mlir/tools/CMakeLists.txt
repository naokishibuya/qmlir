llvm_map_components_to_libnames(llvm_libs Support)

add_executable(quantum-opt
  quantum-opt.cpp
)

llvm_update_compile_flags(quantum-opt)

target_link_libraries(quantum-opt PRIVATE
  MLIRQuantumDialect
  MLIRQuantumPasses
  MLIRIR
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSupport
  ${llvm_libs}
)

mlir_check_all_link_libraries(quantum-opt)
