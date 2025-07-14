# Quantum MLIR Dialect and Gate Cancellation Pass (Experimental)

## Project Description

This repository contains the source files for a **minimal Quantum Dialect** in MLIR and an experimental **C++ optimization pass** that cancels redundant gate sequences.

It demonstrates key compiler concepts for quantum IR:

* **IR Design:** Defining custom quantum gate operations
* **Pass Development:** Writing a C++ pass to analyze and transform IR
* **Optimization:** Simplifying gate sequences (e.g., cancelling `X; X`)

## Features

* Custom MLIR dialect (**QuantumDialect**) with basic operations:

  * `quantum.x`
  * `quantum.h`
  * `quantum.cx`
  * `quantum.alloc`
* C++ pass that detects consecutive `quantum.x` gates on the same qubit and removes them
* Example before/after transformations

## Example

**Input MLIR:**

```
%q = quantum.alloc
quantum.x %q
quantum.x %q
quantum.h %q
```

**After optimization:**

```
%q = quantum.alloc
quantum.h %q
```

## How to Use These Sources

This repository is designed to be **added to an LLVM/MLIR source tree**.

### 1. Clone LLVM Project

```bash
git clone https://github.com/llvm/llvm-project.git
```

### 2. Copy This Folder

This repository contains **only** the dialect and pass source files, intended for integration into LLVM/MLIR.

```bash
cp -rp mlir <path-to-your-llvm-project>/
```
### 3. Update CMake

Add the Quantum dialect and pass to MLIR’s build.

In llvm-project's `mlir/lib/Dialect/CMakeLists.txt`, add:

```cmake
add_subdirectory(Quantum)
```

### 4. Build LLVM/MLIR

```bash
mkdir build
cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release
ninja
```

For MLIR setup instructions, see [MLIR's official documentation](https://mlir.llvm.org/docs/).

### 5. Run the Pass

An example input file is provided in the `examples/` folder. You can run the pass on this file to see the optimization in action.

```bash
bin/mlir-opt --quantum-cancel-x examples/test_input.mlir
```

**Expected output:**

```
%q = quantum.alloc
```
