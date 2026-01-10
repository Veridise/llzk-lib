// Simple tests demonstrating required overflow semantics attribute on cast ops.
// These are small MLIR snippets you can use to test parsing/tablegen output.

module {
  // integer -> felt with truncation (wrap/truncate)
  %i = arith.constant 42 : i32
  %f = cast.tofelt %i {overflow = "trunc"} : i32

  // felt -> int with saturation
  %fe = felt.const 999999
  %i2 = cast.toint %fe {overflow = "sat"} : i8

  // felt -> index with truncation
  %idx = cast.toindex %fe {overflow = "trunc"}
}