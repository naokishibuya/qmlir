#!/bin/bash
# Format all C++ files in the quantum compiler project

set -e

echo "Formatting C++ files with clang-format..."

# Check if clang-format is available
if ! command -v clang-format &> /dev/null; then
    echo "Error: clang-format not found. Please install it:"
    echo "  Ubuntu/Debian: sudo apt install clang-format"
    echo "  macOS: brew install clang-format"
    exit 1
fi

# Find all C++ files and format them
find . -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.c" -o -name "*.cc" | \
  grep -E "(mlir/|qmlir/)" | \
  grep -v build/ | \
  xargs clang-format -i

echo "Done! All C++ files have been formatted."
