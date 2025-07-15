add_llvm_executable(test-quantum-dialect
  test-quantum.cpp
)

target_link_libraries(test-quantum-dialect
  PRIVATE
  MLIRQuantum
  MLIRFuncDialect
  MLIRPass
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)
