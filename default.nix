{ nixpkgs ? import ./nix/nixpkgs.nix
}:

with nixpkgs;

let filterSource = with lib; builtins.filterSource (path: type:
      type != "unknown" &&
      baseNameOf path != "target" &&
      baseNameOf path != "result" &&
      baseNameOf path != ".git" &&
      (baseNameOf path == "build" -> type != "directory") &&
      (baseNameOf path == "nix" -> type != "directory")
    );
in
naersk.buildPackage {
  root = filterSource ./.;

  LIBCLANG_PATH = "${llvmPackages.libclang}/lib";
  ONNXRUNTIME_LIB_DIR = "${onnxruntime}/lib";
  ONNXRUNTIME_INCLUDE_DIR = "${onnxruntime.dev}/include/onnxruntime/core/session:${musl.dev}/include";

  doDoc = true;
  doCheck = true;
}
