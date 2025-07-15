module {
  func.func @bell_state() {
    %q0 = "quantum.alloc"() : () -> i32
    %q1 = "quantum.alloc"() : () -> i32
    "quantum.h"(%q0) : (i32) -> ()
    "quantum.cx"(%q0, %q1) : (i32, i32) -> ()
    return
  }
}
