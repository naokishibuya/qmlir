module {
  func.func @test_quantum_generic() {
    %q = "quantum.alloc"() : () -> index
    "quantum.x"(%q) : (index) -> ()
    "quantum.h"(%q) : (index) -> ()
    %q2 = "quantum.alloc"() : () -> index
    "quantum.cx"(%q, %q2) : (index, index) -> ()
    return
  }
}
