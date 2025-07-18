// RUN: test-quantum-dialect %s | FileCheck %s

module {
  func.func @double_x_test() {
    // CHECK: quantum.alloc
    %q = "quantum.alloc"() : () -> i32
    // CHECK: quantum.x
    "quantum.x"(%q) : (i32) -> ()
    // CHECK: quantum.x
    "quantum.x"(%q) : (i32) -> ()
    // CHECK: quantum.h
    "quantum.h"(%q) : (i32) -> ()
    return
  }
}
