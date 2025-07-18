// RUN: test-quantum-dialect %s | FileCheck %s

module {
  func.func @bell_state() {
    // CHECK: quantum.alloc
    %q0 = "quantum.alloc"() : () -> i32
    // CHECK: quantum.alloc
    %q1 = "quantum.alloc"() : () -> i32
    // CHECK: quantum.h
    "quantum.h"(%q0) : (i32) -> ()
    // CHECK: quantum.cx
    "quantum.cx"(%q0, %q1) : (i32, i32) -> ()
    return
  }
}
