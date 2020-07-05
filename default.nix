{ nixpkgs ? import ./nix/nixpkgs.nix
}:

with nixpkgs;

let filterSource = with lib; builtins.filterSource (path: type:
      type != "unknown" &&
      baseNameOf path != "target" &&
      baseNameOf path != "result" &&
      baseNameOf path != ".git" &&
      baseNameOf path != ".gitignore" &&
      baseNameOf path != ".github" &&
      !(hasSuffix ".nix" path) &&
      (baseNameOf path == "build" -> type != "directory") &&
      (baseNameOf path == "nix" -> type != "directory")
    );

    # Version of onnxruntime.dev with include directory set up the same way as
    # a released version from microsoft/onnxruntime GitHub release.
    onnxruntime-headers = stdenv.mkDerivation {
      name = "onnxruntime-headers";
      src = onnxruntime.dev;
      installPhase = ''
        mkdir -p $out/include

        cp include/onnxruntime/core/session/onnxruntime_c_api.h $out/include/onnxruntime_c_api.h
      '';
    };

in naersk.buildPackage {
  root = filterSource ./.;

  LIBCLANG_PATH = "${llvmPackages.libclang}/lib";
  buildInputs = [ onnxruntime onnxruntime-headers ];
  nativeBuildInputs = [ pkg-config clang ];

  # Note: This creates a doc attribute (nested derivation)
  doDoc = true;

  doCheck = true;
}
