{ stdenv, lib, cmake, ninja, mlir, llzk }:

stdenv.mkDerivation {
  pname = "llzk-installcheck";
  version = "1.0.0";

  src = lib.cleanSource ./.;

  buildInputs = [ mlir llzk ];
  nativeBuildInputs = [ cmake ninja ];

  installPhase = ''touch "$out"'';
}
