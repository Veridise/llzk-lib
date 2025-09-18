{ stdenv, lib, cmake, ninja, mlir_pkg, llzk_pkg }:

stdenv.mkDerivation {
  pname = "llzk-installcheck";
  version = "1.0.0";

  src = lib.cleanSource ./.;

  buildInputs = [ mlir_pkg llzk_pkg ];
  nativeBuildInputs = [ cmake ninja ];

  installPhase = ''touch "$out"'';
}
