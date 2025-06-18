// RUN: r1cs-opt %s | FileCheck %s

r1cs.circuit @example inputs (%a: !r1cs.signal, %b: !r1cs.signal, %c: !r1cs.signal) {
  %d = r1cs.def 1 : !r1cs.signal
  %a_l = r1cs.to_linear %a : !r1cs.signal to !r1cs.linear
  r1cs.return %d, %a : !r1cs.signal, !r1cs.signal
}
// CHECK: r1cs.module @example inputs(%a: !r1cs.signal, %b: !r1cs.signal) outputs -> (!r1cs.signal)
