{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils/v1.0.0";

    veridise-pkgs = {
      url = "git+ssh://git@github.com/Veridise/veridise-nix-pkgs.git?ref=main";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  # Custom colored bash prompt
  nixConfig.bash-prompt = ''\[\e[0;32m\][LLZK]\[\e[m\] \[\e[38;5;244m\]\w\[\e[m\] % '';

  outputs = { self, nixpkgs, flake-utils, veridise-pkgs }:
    {
      # First, we define the packages used in this repository/flake
      overlays.default = final: prev: {
        mlirWithPython = final.mlir.override {
          enablePythonBindings = true;
        };

        llzk = final.callPackage ./nix/llzk.nix { clang = final.clang_18; };

        llzkWithPython = final.llzk.override {
          mlir = final.mlirWithPython;
        };

        llzkDebugClang = (final.llzk.override { stdenv = final.clangStdenv; }).overrideAttrs(attrs: {
          cmakeBuildType = "DebWithSans";

          postInstall = ''
            if [ -f test/report.xml ]; then
              mkdir -p $out/artifacts
              echo "-- Copying xUnit report to $out/artifacts/clang-report.xml"
              cp test/report.xml $out/artifacts/clang-report.xml
            fi
          '';
        });
        llzkDebugClangCov = final.llzkDebugClang.overrideAttrs(attrs: {
          postCheck = ''
            MANIFEST=profiles.manifest
            PROFDATA=coverage.profdata
            BINS=bins.lst
            if [[ "$(uname)" == "Darwin" ]]; then
              find bin lib -type f | xargs file | fgrep Mach-O | grep executable | cut -f1 -d: > $BINS
            else
              find bin lib -type f | xargs file | grep ELF | grep executable | cut -f1 -d: > $BINS
            fi
            echo -n "Found profraw files:"
            find test -name "*.profraw" | tee $MANIFEST | wc -l
            cat $MANIFEST
            llvm-profdata merge -sparse -f $MANIFEST -o $PROFDATA
            OBJS=$( (head -n 1 $BINS ; tail -n +2 $BINS | sed -e "s/^/-object /") | xargs)
            # TODO HTML reports
            llvm-cov report $OBJS -instr-profile $PROFDATA > cov-summary.txt
            echo =========== COVERAGE SUMMARY =================
            cat cov-summary.txt
            echo ==============================================
            llvm-cov export -format=lcov -instr-profile $PROFDATA $OBJS > report.lcov
            rm -rf $MANIFEST $PROFDATA $BINS
          '';

          postInstall = ''
            mkdir -p $out/artifacts/
            echo "-- Copying coverage summary to $out/artifacts/cov-summary.txt"
            cp cov-summary.txt $out/artifacts/
            echo "-- Copying lcov report to $out/artifacts/report.lcov"
            cp report.lcov $out/artifacts/
            if [ -f test/report.xml ]; then
              echo "-- Copying xUnit report to $out/artifacts/clang-report.xml"
              cp test/report.xml $out/artifacts/clang-report.xml
            fi
          '';
        });
        llzkDebugGCC = (final.llzk.override { stdenv = final.gccStdenv; }).overrideAttrs(attrs: {
          cmakeBuildType = "DebWithSans";

          postInstall = ''
            if [ -f test/report.xml ]; then
              mkdir -p $out/artifacts
              echo "-- Copying xUnit report to $out/artifacts/gcc-report.xml"
              cp test/report.xml $out/artifacts/gcc-report.xml
            fi
          '';
        });

        ccacheStdenv = prev.ccacheStdenv.override {
          extraConfig = ''
            export CCACHE_DIR=/tmp/ccache
            export CCACHE_UMASK=007
            export CCACHE_COMPRESS=1
          '';
        };

        # The default shell is used for LLZK development.
        # Because `nix develop` is used to set up a dev shell for a given
        # derivation, we just need to extend the llzk derivation with any
        # extra tools we need.
        devShellBase = { pkgs, llzkEnv ? final.llzk, ... }: {
          shell = llzkEnv.overrideAttrs (old: {
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
        };
      };
    } //
    (flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;

          overlays = [
            self.overlays.default
            veridise-pkgs.overlays.default
          ];
        };
      in
      {
        # Now, we can define the actual outputs of the flake
        packages = flake-utils.lib.flattenTree {
          # Copy the packages from the overlay.
          inherit (pkgs) llzk llzkWithPython;

          # For debug purposes, expose the MLIR/LLVM packages.
          inherit (pkgs) libllvm llvm mlir mlirWithPython;

          default = pkgs.llzk;
          debugClang = pkgs.llzkDebugClang;
          debugClangCov = pkgs.llzkDebugClangCov;
          debugGCC = pkgs.llzkDebugGCC;
        };

        checks = flake-utils.lib.flattenTree {
          llzkInstallCheck = pkgs.callPackage ./nix/llzk-installcheck { };
        };

        devShells = flake-utils.lib.flattenTree {
          default = (pkgs.devShellBase pkgs).shell.overrideAttrs (_: {
            # Use Debug by default so assertions are enabled by default.
            cmakeBuildType = "Debug";
          });
          debugClang = _: (pkgs.devShellBase pkgs pkgs.llzkDebugClang).shell;
          debugGCC = _: (pkgs.devShellBase pkgs pkgs.llzkDebugGCC).shell;

          llvm = pkgs.mkShell {
            buildInputs = [ pkgs.libllvm.dev ];
          };
        };
      }
    ));
}
