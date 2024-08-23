{
  inputs = {
    flake-utils = {
      url = "github:numtide/flake-utils/v1.0.0";
    };
  };

  # Custom colored bash prompt
  nixConfig.bash-prompt = ''\[\e[0;32m\][ZKIR]\[\e[m\] \[\e[38;5;244m\]\w\[\e[m\] % '';

  outputs = { self, nixpkgs, flake-utils }:
    {
      # First, we define the packages used in this repository/flake
      overlays.default = final: prev: {
        # Use a custom version of LLVM to
        # 1) enable some features we need
        # 2) speed up the build
        # 3) add MLIR (which is not packaged in nixpkgs yet)
        zkir_llvm =
          let
            lpkgs = final.llvmPackages_18;
            tools = lpkgs.tools.extend (tpkgs: tpkgsOld: {
              libllvm = (tpkgsOld.libllvm.override ({
                # Skip tests since they take a long time to build and run
                doCheck = false;
              })).overrideAttrs (attrs: {
                cmakeFlags = attrs.cmakeFlags ++ [
                  # Skip irrelevant targets
                  "-DLLVM_TARGETS_TO_BUILD=host"
                  "-DLLVM_INCLUDE_BENCHMARKS=OFF"
                  "-DLLVM_INCLUDE_EXAMPLES=OFF"
                  "-DLLVM_INCLUDE_TESTS=OFF"
                  # Need the following to enable exceptions
                  "-DLLVM_ENABLE_EH=ON"
                  # Assertions are very useful for debugging
                  "-DLLVM_ENABLE_ASSERTIONS=ON"
                ];
              });

              mlir = final.callPackage ./nix/mlir/default.nix {
                inherit (tpkgs.libllvm) monorepoSrc version;
                buildLlvmTools = final.llvmLocal;
                llvm_meta = lpkgs.libllvm.meta;
                inherit (tpkgs) libllvm;

                debugVersion = true;

                # stdenv = final.ccacheStdenv;
                #
                # Unfortunately, ccache is currently broken due to
                # https://github.com/NixOS/nixpkgs/issues/119779
              };

              mlirWithPython = final.mlir.override {
                enablePythonBindings = true;
              };
            });
          in
            lpkgs // { inherit tools; } // tools;

        inherit (final.zkir_llvm) mlir mlirWithPython;

        zkir = final.callPackage ./nix/zkir.nix {};
        zkirWithPython = final.zkir.override {
          mlir = final.mlirWithPython;
        };
        zkirDebugClang = (final.zkir.override { stdenv = final.clangStdenv; }).overrideAttrs(attrs: {
          cmakeBuildType = "DebWithSans";
        });
        zkirDebugGCC = (final.zkir.override { stdenv = final.gccStdenv; }).overrideAttrs(attrs: {
          cmakeBuildType = "DebWithSans";
        });

        ccacheStdenv = prev.ccacheStdenv.override {
          extraConfig = ''
            export CCACHE_DIR=/tmp/ccache
            export CCACHE_UMASK=007
            export CCACHE_COMPRESS=1
          '';
        };
      };
    } //
    (flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ self.overlays.default ];
        };
      in
      {
        # Now, we can define the actual outputs of the flake

        packages = flake-utils.lib.flattenTree {
          # Copy the packages from the overlay.
          inherit (pkgs) zkir zkirWithPython;

          # For debug purposes, expose the MLIR/LLVM packages.
          inherit (pkgs.zkir_llvm.tools) libllvm;
          inherit (pkgs.zkir_llvm.tools) llvm;
          inherit (pkgs) mlir mlirWithPython;

          default = pkgs.zkir;
          debugClang = pkgs.zkirDebugClang;
          debugGCC = pkgs.zkirDebugGCC;
        };

        devShells = flake-utils.lib.flattenTree {
          # The default shell is used for ZKIR development.
          # Because `nix develop` is used to set up a dev shell for a given
          # derivation, we just need to extend the zkir derivation with any
          # extra tools we need.
          default = pkgs.zkir.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ (with pkgs; [
              doxygen

              # clang-tidy and clang-format
              clang-tools_18

              # git-clang-format
              libclang.python
            ]);

            # Use Debug by default so assertions are enabled by default.
            cmakeBuildType = "Debug";

            shellHook = ''
              # needed to get accurate compile_commands.json
              export CXXFLAGS="$NIX_CFLAGS_COMPILE"

              # Add binary dir to PATH for convenience
              export PATH="$PWD"/build/bin:"$PATH"

              # TODO: only enable if python bindings enabled
              export PYTHONPATH="$PYTHONPATH":"$PWD"/build/python
            '';
          });

          debugClang = pkgs.zkirDebugClang.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ (with pkgs; [
              doxygen

              # clang-tidy and clang-format
              clang-tools_18

              # git-clang-format
              libclang.python
            ]);

            shellHook = ''
              # needed to get accurate compile_commands.json
              export CXXFLAGS="$NIX_CFLAGS_COMPILE"

              # Add binary dir to PATH for convenience
              export PATH="$PWD"/build/bin:"$PATH"

              # TODO: only enable if python bindings enabled
              export PYTHONPATH="$PYTHONPATH":"$PWD"/build/python
            '';
          });

          debugGCC = pkgs.zkirDebugGCC.overrideAttrs (old: {
            nativeBuildInputs = old.nativeBuildInputs ++ (with pkgs; [
              doxygen

              # clang-tidy and clang-format
              clang-tools_18

              # git-clang-format
              libclang.python
            ]);

            shellHook = ''
              # needed to get accurate compile_commands.json
              export CXXFLAGS="$NIX_CFLAGS_COMPILE"

              # Add binary dir to PATH for convenience
              export PATH="$PWD"/build/bin:"$PATH"

              # TODO: only enable if python bindings enabled
              export PYTHONPATH="$PYTHONPATH":"$PWD"/build/python
            '';
          });


          llvm = pkgs.mkShell {
            buildInputs = [ pkgs.zkir_llvm.tools.libllvm.dev ];
          };
        };
      }
    ));
}
