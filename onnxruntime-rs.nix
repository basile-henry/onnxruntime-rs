{ nixpkgs ? import ./nix/nixpkgs.nix
, onnxruntime ? null # onnxruntime library needed for tests
, onnxruntime-headers ? "${nixpkgs.onnxruntime.dev}/include/onnxruntime/core/session"
}:

with nixpkgs;

let filterSource = src: lib.cleanSourceWith {
      inherit src;
      filter = path: type:
        baseNameOf path != "target" && lib.cleanSourceFilter path type;
    };
in
naersk.buildPackage ({
  root = filterSource ./.;
  doDoc = true;

  LIBCLANG_PATH = "${llvmPackages.libclang}/lib";
  C_INCLUDE_PATH = "${onnxruntime-headers}:${musl.dev}/include";

} // lib.optionalAttrs (onnxruntime != null) {
  buildInputs = [ onnxruntime ];
  doCheck = true;
})
