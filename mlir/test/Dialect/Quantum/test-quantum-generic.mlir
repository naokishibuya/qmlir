// RUN: test-quantum-dialect %s | FileCheck %s

module {
  func.func @test_quantum_generic() {
    // CHECK: quantum.alloc
    %q = "quantum.alloc"() : () -> index
    // CHECK: quantum.x
    "quantum.x"(%q) : (index) -> ()
    // CHECK: quantum.h
    "quantum.h"(%q) : (index) -> ()
    // CHECK: quantum.alloc
    %q2 = "quantum.alloc"() : () -> index
    // CHECK: quantum.cx
    "quantum.cx"(%q, %q2) : (index, index) -> ()
    return
  }
}
