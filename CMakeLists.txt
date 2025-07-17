cmake_minimum_required(VERSION 3.13)
project(QuantumCompiler VERSION 0.1.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow user to specify LLVM path
if(NOT DEFINED LLVM_DIR)
  set(LLVM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/build/lib/cmake/llvm")
endif()

if(NOT DEFINED MLIR_DIR)
  set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/build/lib/cmake/mlir")
endif()

# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Set the LLVM header and library paths
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include the necessary directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/backend/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/backend/include)

# Add compile definitions
add_definitions(${LLVM_DEFINITIONS})

# Configure the header to pass some CMake settings to the source code
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/backend/include/mlir/Dialect/Quantum/IR/QuantumConfig.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/backend/include/mlir/Dialect/Quantum/IR/QuantumConfig.h
)

# Add component subdirectories
add_subdirectory(backend/include/mlir/Dialect/Quantum)
add_subdirectory(backend/lib/Dialect/Quantum)  
add_subdirectory(backend/tools)
