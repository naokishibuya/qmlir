// RUN: cat %s | FileCheck --check-prefix=BEFORE %s
// RUN: quantum-opt --quantum-cancel-self-inverse %s | FileCheck --check-prefix=AFTER %s

module {
  func.func @double_x_test() {
    // BEFORE: quantum.alloc
    // BEFORE: quantum.x
    // BEFORE: quantum.x
    // BEFORE: quantum.h

    // AFTER: quantum.alloc
    // AFTER-NOT: quantum.x
    // AFTER: quantum.h

    %q = "quantum.alloc"() : () -> i32
    "quantum.x"(%q) : (i32) -> ()
    "quantum.x"(%q) : (i32) -> ()
    "quantum.h"(%q) : (i32) -> ()
    return
  }
}
